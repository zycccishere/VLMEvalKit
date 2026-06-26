#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("VLMEVAL_VLM_MINIMAL_IMPORT", "1")
os.environ.setdefault("VLMEVAL_USE_QWEN_MINIMAL_CONFIG", "1")

from qwen25vl_image2_probe import (  # noqa: E402
    build_base_content,
    build_inputs,
    build_replayed_content,
    find_image_spans,
    load_model_and_processor,
    parse_attention_layers,
    resolve_input_device,
    sanitize_single_process_env,
    set_seed,
    tensor_to_device,
)
from vlmeval.attention import (  # noqa: E402
    AttentionFullMapSpec,
    AttentionMatrixSpec,
    PositionGroup,
    QwenPrefillAttentionTracer,
    get_language_model_layers,
)
from vlmeval.cross_image_flow_v2 import extract_image_grid_meta, image_token_table  # noqa: E402
from vlmeval.dataset import build_dataset  # noqa: E402
from vlmeval.vlm.qwen2_vl.replay_prompt_template import (  # noqa: E402
    PROMPT_TEMPLATE_DIRECTLY_ANSWER,
    PROMPT_TEMPLATE_IDENTITY,
)


KEY_GROUP_ORDER = ["i1", "q1", "i2", "q2", "other", "special"]
QUERY_GROUP_ORDER = ["i2", "q2"]
MATRIX_SPECS = [
    AttentionMatrixSpec("i2_to_i1", "i2", "i1"),
    AttentionMatrixSpec("i2_to_q1", "i2", "q1"),
    AttentionMatrixSpec("q2_to_q1", "q2", "q1"),
    AttentionMatrixSpec("q2_to_i1", "q2", "i1"),
    AttentionMatrixSpec("q2_to_i2", "q2", "i2"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize Qwen2.5-VL prefill attention in the IQIQ "
            "(image_text_image_text) replay setting."
        )
    )
    parser.add_argument("--model-path", default="/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset", default="AI2D_TEST")
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--mode", default="image_text_image_text")
    parser.add_argument(
        "--policy",
        default=PROMPT_TEMPLATE_IDENTITY,
        choices=[PROMPT_TEMPLATE_IDENTITY, PROMPT_TEMPLATE_DIRECTLY_ANSWER],
    )
    parser.add_argument("--template-on-last-replay-text", action="store_true")
    parser.add_argument("--attn-layers", default="last4")
    parser.add_argument(
        "--heads",
        default="0,1,2,3",
        help="Heads to render in detailed figures: all, firstN, last, or comma-separated indices.",
    )
    parser.add_argument(
        "--full-vtvt-map",
        action="store_true",
        help="Save complete V/T/V/T query-by-key maps for selected heads/layers/samples.",
    )
    parser.add_argument(
        "--full-map-heads",
        default="",
        help="Heads for full VTVT maps. Defaults to --heads when empty.",
    )
    parser.add_argument(
        "--full-map-samples",
        nargs="*",
        type=int,
        default=[],
        help="Sample indices for full VTVT maps. Defaults to all selected indices when empty.",
    )
    parser.add_argument(
        "--full-map-layers",
        default="",
        help="Layers for full VTVT maps. Defaults to --attn-layers when empty.",
    )
    parser.add_argument("--full-map-plot-max-tokens", type=int, default=1600)
    parser.add_argument("--max-full-matrix-elements", type=int, default=4_000_000)
    parser.add_argument("--max-token-label-chars", type=int, default=18)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)
    return parser


def json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def safe_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def row_summary(row: pd.Series) -> dict[str, Any]:
    preferred = [
        "index",
        "question",
        "answer",
        "category",
        "split",
        "image",
        "A",
        "B",
        "C",
        "D",
        "E",
    ]
    out: dict[str, Any] = {}
    for key in preferred:
        if key in row:
            out[key] = safe_scalar(row[key])
    return out


def decode_token_labels(tokenizer, input_ids: list[int], positions: list[int], max_chars: int) -> list[str]:
    labels = []
    for pos in positions:
        token = tokenizer.decode(
            [int(input_ids[pos])],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        token = token.replace("\n", "\\n")
        if len(token) > max_chars:
            token = token[: max(1, max_chars - 3)] + "..."
        labels.append(f"{pos}:{token}")
    return labels


def find_subsequence(candidate_ids: list[int], pattern: list[int]) -> int:
    if not pattern or len(pattern) > len(candidate_ids):
        return -1
    first = pattern[0]
    limit = len(candidate_ids) - len(pattern) + 1
    for start in range(limit):
        if candidate_ids[start] != first:
            continue
        if candidate_ids[start : start + len(pattern)] == pattern:
            return start
    return -1


def derive_iqiq_groups(
    *,
    input_ids: list[int],
    image_spans: list[Any],
    special_token_ids: set[int],
) -> list[PositionGroup]:
    if len(image_spans) != 2:
        raise ValueError(f"IQIQ tracing expects exactly two image spans, got {len(image_spans)}.")
    i1_positions = list(range(int(image_spans[0].start), int(image_spans[0].end) + 1))
    i2_positions = list(range(int(image_spans[1].start), int(image_spans[1].end) + 1))
    image_positions = set(i1_positions) | set(i2_positions)

    q1_positions = [
        pos
        for pos in range(int(image_spans[0].end) + 1, int(image_spans[1].start))
        if input_ids[pos] not in special_token_ids
    ]
    q1_ids = [int(input_ids[pos]) for pos in q1_positions]

    q2_candidates = [
        pos
        for pos in range(int(image_spans[1].end) + 1, len(input_ids))
        if input_ids[pos] not in special_token_ids and pos not in image_positions
    ]
    q2_candidate_ids = [int(input_ids[pos]) for pos in q2_candidates]
    q2_start = find_subsequence(q2_candidate_ids, q1_ids)
    if q2_start < 0:
        raise ValueError(
            "Failed to identify Q2 as an exact token replay of Q1 after image2; "
            "check the replay mode or prompt template."
        )
    q2_positions = q2_candidates[q2_start : q2_start + len(q1_positions)]

    covered = set(i1_positions) | set(i2_positions) | set(q1_positions) | set(q2_positions)
    special_positions = [
        pos
        for pos, token_id in enumerate(input_ids)
        if pos not in covered and int(token_id) in special_token_ids
    ]
    other_positions = [
        pos
        for pos, token_id in enumerate(input_ids)
        if pos not in covered and pos not in special_positions and int(token_id) not in special_token_ids
    ]

    return [
        PositionGroup.from_positions("i1", i1_positions, kind="image"),
        PositionGroup.from_positions("q1", q1_positions, kind="text"),
        PositionGroup.from_positions("i2", i2_positions, kind="image"),
        PositionGroup.from_positions("q2", q2_positions, kind="text"),
        PositionGroup.from_positions("other", other_positions, kind="text"),
        PositionGroup.from_positions("special", special_positions, kind="special"),
    ]


def parse_head_selection(spec: str, head_count: int) -> list[int]:
    raw = (spec or "0,1,2,3").strip().lower()
    if raw == "all":
        return list(range(head_count))
    if raw == "last":
        return [head_count - 1]
    if raw.startswith("first") and raw[5:].isdigit():
        return list(range(min(head_count, int(raw[5:]))))
    heads: list[int] = []
    seen: set[int] = set()
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        idx = head_count - 1 if part == "last" else int(part)
        if idx < 0:
            idx += head_count
        if idx < 0 or idx >= head_count:
            raise ValueError(f"Head index out of range for spec={spec!r}: {part}")
        if idx not in seen:
            heads.append(idx)
            seen.add(idx)
    if not heads:
        raise ValueError(f"Empty head selection: {spec!r}")
    return heads


def group_dict(groups: list[PositionGroup]) -> dict[str, PositionGroup]:
    return {group.name: group for group in groups}


def span_dict(span: Any) -> dict[str, int]:
    return {
        "name": str(span.name),
        "start": int(span.start),
        "end": int(span.end),
        "length": int(span.length),
    }


def matrix_npz_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = {
        "query_positions": np.asarray(payload["query_positions"], dtype=np.int32),
        "key_positions": np.asarray(payload["key_positions"], dtype=np.int32),
        "query_mean": np.asarray(payload["query_mean"], dtype=np.float16),
        "head_mass": np.asarray(payload["head_mass"], dtype=np.float32),
        "matrix_shape": np.asarray(payload["matrix_shape"], dtype=np.int32),
        "matrix_stored": np.asarray([1 if payload["matrix_stored"] else 0], dtype=np.int8),
    }
    if payload.get("matrix_stored") and "matrix" in payload:
        out["matrix"] = np.asarray(payload["matrix"], dtype=np.float16)
    return out


def write_group_rows(
    *,
    rows: list[dict[str, Any]],
    sample_index: int,
    sample_id: str,
    record: dict[str, Any],
) -> None:
    for group_row in record["group_masses"]:
        mass = np.asarray(group_row["mass_by_head"], dtype=np.float32)
        std = np.asarray(group_row["std_by_head"], dtype=np.float32)
        for head_idx, (mass_value, std_value) in enumerate(zip(mass.tolist(), std.tolist())):
            rows.append(
                {
                    "sample_index": int(sample_index),
                    "sample_id": sample_id,
                    "layer": int(record["layer"]),
                    "head": int(head_idx),
                    "query_group": group_row["query_group"],
                    "key_group": group_row["key_group"],
                    "mass": float(mass_value),
                    "mass_query_std": float(std_value),
                    "query_count": int(group_row["query_count"]),
                    "key_count": int(group_row["key_count"]),
                    "q_len": int(record["q_len"]),
                    "kv_len": int(record["kv_len"]),
                }
            )


def plot_group_mass_heatmap(
    *,
    df: pd.DataFrame,
    sample_index: int,
    layer: int,
    query_group: str,
    heads: list[int],
    out_path: Path,
) -> None:
    subset = df[
        (df["sample_index"] == sample_index)
        & (df["layer"] == layer)
        & (df["query_group"] == query_group)
        & (df["head"].isin(heads))
    ]
    if subset.empty:
        return
    matrix = []
    ylabels = []
    for head in heads:
        head_df = subset[subset["head"] == head]
        if head_df.empty:
            continue
        values = []
        for key_group in KEY_GROUP_ORDER:
            cell = head_df[head_df["key_group"] == key_group]
            values.append(float(cell["mass"].iloc[0]) if not cell.empty else np.nan)
        matrix.append(values)
        ylabels.append(f"H{head}")
    if not matrix:
        return
    arr = np.asarray(matrix, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8.2, max(2.8, 0.42 * len(ylabels) + 1.6)))
    im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(1e-6, float(np.nanmax(arr))))
    ax.set_xticks(np.arange(len(KEY_GROUP_ORDER)))
    ax.set_xticklabels(KEY_GROUP_ORDER)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title(f"sample {sample_index} layer {layer} query {query_group}")
    ax.set_xlabel("key group")
    ax.set_ylabel("attention head")
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            value = arr[y, x]
            if np.isfinite(value):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax, label="mean attention mass")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def image_keymap(payload: dict[str, Any], meta: Any, head: int) -> np.ndarray:
    values = np.asarray(payload["query_mean"], dtype=np.float32)[head]
    expected = int(meta.temporal) * int(meta.llm_grid_h) * int(meta.llm_grid_w)
    if values.size != expected:
        raise ValueError(
            f"Key-map size mismatch for {payload['key_group']}: values={values.size}, expected={expected}."
        )
    arr = values.reshape(int(meta.temporal), int(meta.llm_grid_h), int(meta.llm_grid_w))
    return arr.sum(axis=0)


def plot_image_keymap(
    *,
    payload: dict[str, Any],
    meta: Any,
    sample_index: int,
    layer: int,
    head: int,
    out_path: Path,
) -> None:
    keymap = image_keymap(payload, meta, head)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(keymap, cmap="magma")
    ax.set_title(f"{payload['query_group']} -> {payload['key_group']} S{sample_index} L{layer} H{head}")
    ax.set_xlabel("image-token col")
    ax.set_ylabel("image-token row")
    fig.colorbar(im, ax=ax, label="query-mean attention")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_text_matrix(
    *,
    payload: dict[str, Any],
    input_ids: list[int],
    tokenizer,
    sample_index: int,
    layer: int,
    head: int,
    max_token_label_chars: int,
    out_path: Path,
) -> None:
    if not payload.get("matrix_stored") or "matrix" not in payload:
        return
    matrix = np.asarray(payload["matrix"], dtype=np.float32)[head]
    q_labels = decode_token_labels(tokenizer, input_ids, payload["query_positions"], max_token_label_chars)
    k_labels = decode_token_labels(tokenizer, input_ids, payload["key_positions"], max_token_label_chars)
    fig_w = min(16, max(6, 0.32 * len(k_labels)))
    fig_h = min(14, max(5, 0.28 * len(q_labels)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(f"{payload['query_group']} -> {payload['key_group']} S{sample_index} L{layer} H{head}")
    tick_stride_x = max(1, int(np.ceil(len(k_labels) / 28)))
    tick_stride_y = max(1, int(np.ceil(len(q_labels) / 24)))
    ax.set_xticks(np.arange(0, len(k_labels), tick_stride_x))
    ax.set_xticklabels(k_labels[::tick_stride_x], rotation=65, ha="right", fontsize=7)
    ax.set_yticks(np.arange(0, len(q_labels), tick_stride_y))
    ax.set_yticklabels(q_labels[::tick_stride_y], fontsize=7)
    ax.set_xlabel(f"key tokens: {payload['key_group']}")
    ax.set_ylabel(f"query tokens: {payload['query_group']}")
    fig.colorbar(im, ax=ax, label="attention")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_layer_summary(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    grouped = df.groupby(["layer", "query_group", "key_group"], as_index=False)["mass"].mean()
    fig, axes = plt.subplots(1, len(QUERY_GROUP_ORDER), figsize=(12, 4.2), sharey=True)
    if len(QUERY_GROUP_ORDER) == 1:
        axes = [axes]
    for ax, query_group in zip(axes, QUERY_GROUP_ORDER):
        subset = grouped[grouped["query_group"] == query_group]
        for key_group in KEY_GROUP_ORDER:
            line = subset[subset["key_group"] == key_group]
            if line.empty:
                continue
            ax.plot(line["layer"], line["mass"], marker="o", linewidth=1.6, label=key_group)
        ax.set_title(f"query {query_group}")
        ax.set_xlabel("layer")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean attention mass")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def partition_bounds(size: int, max_bins: int) -> list[tuple[int, int]]:
    if size <= max_bins:
        return [(idx, idx + 1) for idx in range(size)]
    bounds = []
    for idx in range(max_bins):
        start = int(round(size * idx / max_bins))
        end = int(round(size * (idx + 1) / max_bins))
        bounds.append((start, max(start + 1, end)))
    return bounds


def downsample_matrix_for_plot(matrix: np.ndarray, max_tokens: int) -> tuple[np.ndarray, float, float]:
    rows, cols = matrix.shape
    if max(rows, cols) <= max_tokens:
        return matrix, 1.0, 1.0
    row_bins = min(rows, max_tokens)
    col_bins = min(cols, max_tokens)
    row_bounds = partition_bounds(rows, row_bins)
    col_bounds = partition_bounds(cols, col_bins)
    out = np.zeros((row_bins, col_bins), dtype=np.float32)
    for row_idx, (row_start, row_end) in enumerate(row_bounds):
        row_block = matrix[row_start:row_end]
        for col_idx, (col_start, col_end) in enumerate(col_bounds):
            out[row_idx, col_idx] = float(row_block[:, col_start:col_end].mean())
    return out, row_bins / float(rows), col_bins / float(cols)


def segment_centers(segments: list[dict[str, Any]], scale: float) -> tuple[list[float], list[str]]:
    centers = []
    labels = []
    for segment in segments:
        start = int(segment["start"])
        end = int(segment["end"])
        if end <= start:
            continue
        centers.append(((start + end - 1) / 2.0) * scale)
        labels.append(str(segment["group"]))
    return centers, labels


def draw_segment_boundaries(ax, segments: list[dict[str, Any]], scale: float, *, axis: str) -> None:
    for segment in segments:
        end = int(segment["end"])
        if end <= 0:
            continue
        pos = end * scale - 0.5
        if axis == "x":
            ax.axvline(pos, color="white", linewidth=0.7, alpha=0.75)
        else:
            ax.axhline(pos, color="white", linewidth=0.7, alpha=0.75)


def plot_full_vtvt_map(
    *,
    matrix: np.ndarray,
    query_segments: list[dict[str, Any]],
    key_segments: list[dict[str, Any]],
    sample_index: int,
    layer: int,
    head: int,
    max_tokens: int,
    out_path: Path,
) -> None:
    plot_matrix, row_scale, col_scale = downsample_matrix_for_plot(matrix.astype(np.float32), max_tokens)
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(plot_matrix, aspect="auto", cmap="viridis")
    q_centers, q_labels = segment_centers(query_segments, row_scale)
    k_centers, k_labels = segment_centers(key_segments, col_scale)
    ax.set_xticks(k_centers)
    ax.set_xticklabels(k_labels)
    ax.set_yticks(q_centers)
    ax.set_yticklabels(q_labels)
    draw_segment_boundaries(ax, key_segments, col_scale, axis="x")
    draw_segment_boundaries(ax, query_segments, row_scale, axis="y")
    ax.set_title(f"Full VTVT prefill attention S{sample_index} L{layer} H{head}")
    ax.set_xlabel("key groups")
    ax.set_ylabel("query groups")
    fig.colorbar(im, ax=ax, label="attention")
    fig.tight_layout()
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    sanitize_single_process_env()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_matrices"
    full_map_dir = output_dir / "full_maps"
    fig_dir = output_dir / "figures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    full_map_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    processor, model = load_model_and_processor(args.model_path, args.device)
    input_device = resolve_input_device(model, args.device)
    layers = get_language_model_layers(model)
    selected_layers = parse_attention_layers(args.attn_layers, len(layers))
    full_map_layers = (
        parse_attention_layers(args.full_map_layers, len(layers))
        if args.full_map_layers.strip()
        else list(selected_layers)
    )
    attn_modules = {layer_idx: layers[layer_idx].self_attn for layer_idx in selected_layers}
    head_count_for_args = int(attn_modules[selected_layers[0]].num_heads)
    full_map_heads = parse_head_selection(
        args.full_map_heads if args.full_map_heads.strip() else args.heads,
        head_count_for_args,
    )
    full_map_sample_set = set(args.full_map_samples or args.indices)
    full_map_specs = [
        AttentionFullMapSpec.from_groups(
            "vtvt_full",
            ["i1", "q1", "i2", "q2"],
            ["i1", "q1", "i2", "q2"],
            heads=full_map_heads,
        )
    ] if args.full_vtvt_map else []
    tracer = QwenPrefillAttentionTracer(
        attn_modules,
        mass_query_groups=QUERY_GROUP_ORDER,
        mass_key_groups=KEY_GROUP_ORDER,
        matrix_specs=MATRIX_SPECS,
        full_map_specs=full_map_specs,
        max_full_matrix_elements=args.max_full_matrix_elements,
    )
    tracer.patch()

    dataset = build_dataset(args.dataset)
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    special_token_ids = set(processor.tokenizer.all_special_ids)
    spatial_merge_size = int(model.model.visual.spatial_merge_size)

    samples: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    matrix_manifest: list[dict[str, Any]] = []
    full_map_manifest: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    plot_heads: list[int] | None = None

    try:
        for sample_index in args.indices:
            sample_start = time.perf_counter()
            row = dataset.data.iloc[int(sample_index)]
            base_content = build_base_content(dataset, row)
            content = build_replayed_content(
                base_content,
                args.dataset,
                mode=args.mode,
                policy=args.policy,
                template_on_last_replay_text=args.template_on_last_replay_text,
            )
            _, prompt_text, model_inputs = build_inputs(processor, content)
            input_ids = [int(x) for x in model_inputs["input_ids"][0].tolist()]
            image_spans = find_image_spans(input_ids, image_token_id)
            groups = derive_iqiq_groups(
                input_ids=input_ids,
                image_spans=image_spans,
                special_token_ids=special_token_ids,
            )
            group_by_name = group_dict(groups)
            grid_metas = extract_image_grid_meta(
                model_inputs["image_grid_thw"],
                spatial_merge_size=spatial_merge_size,
            )
            if len(grid_metas) != 2:
                raise ValueError(f"Expected exactly two image grids for IQIQ, got {len(grid_metas)}.")
            for image_name, grid_meta, span in zip(["i1", "i2"], grid_metas, image_spans):
                if int(grid_meta.token_count) != int(span.length):
                    raise ValueError(
                        f"{image_name} token count mismatch: grid={grid_meta.token_count}, span={span.length}."
                    )

            tracer.configure_sample(
                sample_id=str(sample_index),
                groups=groups,
                matrix_specs=MATRIX_SPECS,
                full_map_specs=(
                    full_map_specs
                    if args.full_vtvt_map and int(sample_index) in full_map_sample_set
                    else []
                ),
            )
            tracer.reset()

            model_inputs = tensor_to_device(model_inputs, input_device)
            prefill_start = time.perf_counter()
            with torch.inference_mode():
                outputs = model(
                    **model_inputs,
                    use_cache=False,
                    return_dict=True,
                )
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            prefill_seconds = time.perf_counter() - prefill_start
            records = list(tracer.records)
            if not records:
                raise RuntimeError(f"No prefill attention records captured for sample {sample_index}.")

            head_count = int(np.asarray(records[0]["group_masses"][0]["mass_by_head"]).shape[0])
            if plot_heads is None:
                plot_heads = parse_head_selection(args.heads, head_count)

            sample_matrices = []
            sample_full_maps = []
            for record in records:
                write_group_rows(
                    rows=group_rows,
                    sample_index=int(sample_index),
                    sample_id=str(sample_index),
                    record=record,
                )
                for name, payload in record["matrices"].items():
                    rel_path = Path("raw_matrices") / f"sample{sample_index}_layer{record['layer']}_{name}.npz"
                    np.savez_compressed(output_dir / rel_path, **matrix_npz_payload(payload))
                    item = {
                        "sample_index": int(sample_index),
                        "layer": int(record["layer"]),
                        "name": name,
                        "query_group": payload["query_group"],
                        "key_group": payload["key_group"],
                        "query_count": int(len(payload["query_positions"])),
                        "key_count": int(len(payload["key_positions"])),
                        "matrix_shape": payload["matrix_shape"],
                        "matrix_stored": bool(payload["matrix_stored"]),
                        "path": str(rel_path),
                    }
                    matrix_manifest.append(item)
                    sample_matrices.append(item)
                for name, payload in record.get("full_maps", {}).items():
                    if int(record["layer"]) not in full_map_layers:
                        continue
                    rel_path = Path("full_maps") / f"sample{sample_index}_layer{record['layer']}_{name}.npz"
                    np.savez_compressed(
                        output_dir / rel_path,
                        query_positions=np.asarray(payload["query_positions"], dtype=np.int32),
                        key_positions=np.asarray(payload["key_positions"], dtype=np.int32),
                        query_segments_json=np.asarray(
                            [json.dumps(payload["query_segments"], ensure_ascii=False)]
                        ),
                        key_segments_json=np.asarray(
                            [json.dumps(payload["key_segments"], ensure_ascii=False)]
                        ),
                        heads=np.asarray(payload["heads"], dtype=np.int32),
                        matrix_shape=np.asarray(payload["matrix_shape"], dtype=np.int32),
                        matrix_stored=np.asarray([1 if payload["matrix_stored"] else 0], dtype=np.int8),
                        matrix=np.asarray(payload.get("matrix", np.empty((0,), dtype=np.float16)), dtype=np.float16),
                    )
                    item = {
                        "sample_index": int(sample_index),
                        "layer": int(record["layer"]),
                        "name": name,
                        "query_groups": list(payload["query_groups"]),
                        "key_groups": list(payload["key_groups"]),
                        "heads": [int(head) for head in payload["heads"]],
                        "query_count": int(len(payload["query_positions"])),
                        "key_count": int(len(payload["key_positions"])),
                        "matrix_shape": payload["matrix_shape"],
                        "matrix_stored": bool(payload["matrix_stored"]),
                        "path": str(rel_path),
                    }
                    full_map_manifest.append(item)
                    sample_full_maps.append(item)

            image_tables = {
                "i1": image_token_table(grid_metas[0]),
                "i2": image_token_table(grid_metas[1]),
            }
            group_records = {group.name: group.to_dict() for group in groups}
            sample_summary = {
                "sample_index": int(sample_index),
                "dataset_row": row_summary(row),
                "prompt_text": prompt_text,
                "input_ids": input_ids,
                "input_len": int(len(input_ids)),
                "image_spans": [span_dict(span) for span in image_spans],
                "groups": group_records,
                "q1_token_labels": decode_token_labels(
                    processor.tokenizer,
                    input_ids,
                    list(group_by_name["q1"].positions),
                    args.max_token_label_chars,
                ),
                "q2_token_labels": decode_token_labels(
                    processor.tokenizer,
                    input_ids,
                    list(group_by_name["q2"].positions),
                    args.max_token_label_chars,
                ),
                "image_grids": {
                    "i1": grid_metas[0].to_dict(),
                    "i2": grid_metas[1].to_dict(),
                },
                "image_token_tables": image_tables,
                "matrices": sample_matrices,
                "full_maps": sample_full_maps,
                "prefill_seconds": float(prefill_seconds),
                "seconds": float(time.perf_counter() - sample_start),
            }
            samples.append(sample_summary)
            print(
                json.dumps(
                    {
                        "event": "sample_complete",
                        "sample_index": int(sample_index),
                        "layers": selected_layers,
                        "input_len": len(input_ids),
                        "seconds": sample_summary["seconds"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    finally:
        tracer.restore()

    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(output_dir / "group_masses.csv", index=False)
    if not group_df.empty:
        summary_df = (
            group_df.groupby(["query_group", "key_group", "layer"], as_index=False)
            .agg(
                mean_mass=("mass", "mean"),
                std_mass=("mass", "std"),
                min_mass=("mass", "min"),
                max_mass=("mass", "max"),
                n=("mass", "size"),
            )
        )
        summary_df.to_csv(output_dir / "group_mass_summary.csv", index=False)
        plot_layer_summary(group_df, fig_dir / "group_mass_by_layer.png")

    with (output_dir / "matrix_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_index",
            "layer",
            "name",
            "query_group",
            "key_group",
            "query_count",
            "key_count",
            "matrix_shape",
            "matrix_stored",
            "path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in matrix_manifest:
            item = dict(item)
            item["matrix_shape"] = json.dumps(item["matrix_shape"])
            writer.writerow(item)

    with (output_dir / "full_map_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_index",
            "layer",
            "name",
            "query_groups",
            "key_groups",
            "heads",
            "query_count",
            "key_count",
            "matrix_shape",
            "matrix_stored",
            "path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in full_map_manifest:
            item = dict(item)
            item["query_groups"] = json.dumps(item["query_groups"])
            item["key_groups"] = json.dumps(item["key_groups"])
            item["heads"] = json.dumps(item["heads"])
            item["matrix_shape"] = json.dumps(item["matrix_shape"])
            writer.writerow(item)

    if plot_heads is None:
        plot_heads = []

    for item in full_map_manifest:
        if not item.get("matrix_stored"):
            continue
        with np.load(output_dir / item["path"], allow_pickle=False) as npz:
            matrix = np.asarray(npz["matrix"], dtype=np.float32)
            heads = np.asarray(npz["heads"], dtype=np.int32).tolist()
            query_segments = json.loads(str(npz["query_segments_json"][0]))
            key_segments = json.loads(str(npz["key_segments_json"][0]))
        for head_offset, head in enumerate(heads):
            plot_full_vtvt_map(
                matrix=matrix[head_offset],
                query_segments=query_segments,
                key_segments=key_segments,
                sample_index=int(item["sample_index"]),
                layer=int(item["layer"]),
                head=int(head),
                max_tokens=int(args.full_map_plot_max_tokens),
                out_path=fig_dir
                / f"sample{item['sample_index']}_layer{item['layer']}_head{head}_{item['name']}.png",
            )

    sample_by_index = {int(sample["sample_index"]): sample for sample in samples}
    for sample_index, sample in sample_by_index.items():
        input_ids = [int(x) for x in sample["groups"]["special"]["positions"]]
        del input_ids
        for layer in selected_layers:
            for query_group in QUERY_GROUP_ORDER:
                plot_group_mass_heatmap(
                    df=group_df,
                    sample_index=sample_index,
                    layer=int(layer),
                    query_group=query_group,
                    heads=plot_heads,
                    out_path=fig_dir / f"sample{sample_index}_layer{layer}_{query_group}_group_mass.png",
                )

    for item in matrix_manifest:
        with np.load(output_dir / item["path"]) as npz:
            payload = {
                "query_group": item["query_group"],
                "key_group": item["key_group"],
                "query_positions": npz["query_positions"].tolist(),
                "key_positions": npz["key_positions"].tolist(),
                "query_mean": npz["query_mean"],
                "head_mass": npz["head_mass"],
                "matrix_shape": npz["matrix_shape"].tolist(),
                "matrix_stored": bool(int(npz["matrix_stored"][0])),
            }
            if payload["matrix_stored"] and "matrix" in npz:
                payload["matrix"] = npz["matrix"]
        sample = sample_by_index[int(item["sample_index"])]
        input_ids_for_sample = [int(token) for token in sample["input_ids"]]
        key_group = item["key_group"]
        image_meta = None
        if key_group == "i1":
            image_meta = type("Grid", (), sample["image_grids"]["i1"])()
        elif key_group == "i2":
            image_meta = type("Grid", (), sample["image_grids"]["i2"])()
        for head in plot_heads:
            if image_meta is not None:
                plot_image_keymap(
                    payload=payload,
                    meta=image_meta,
                    sample_index=int(item["sample_index"]),
                    layer=int(item["layer"]),
                    head=int(head),
                    out_path=fig_dir / f"sample{item['sample_index']}_layer{item['layer']}_head{head}_{item['name']}_keymap.png",
                )
            elif item["name"] == "q2_to_q1":
                plot_text_matrix(
                    payload=payload,
                    input_ids=input_ids_for_sample,
                    tokenizer=processor.tokenizer,
                    sample_index=int(item["sample_index"]),
                    layer=int(item["layer"]),
                    head=int(head),
                    max_token_label_chars=int(args.max_token_label_chars),
                    out_path=fig_dir
                    / f"sample{item['sample_index']}_layer{item['layer']}_head{head}_{item['name']}_tokens.png",
                )

    summary = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "indices": [int(x) for x in args.indices],
        "mode": args.mode,
        "policy": args.policy,
        "template_on_last_replay_text": bool(args.template_on_last_replay_text),
        "attn_layers": args.attn_layers,
        "selected_layers": [int(x) for x in selected_layers],
        "heads_for_plots": [int(x) for x in plot_heads],
        "matrix_specs": [spec.to_dict() for spec in MATRIX_SPECS],
        "full_vtvt_map": bool(args.full_vtvt_map),
        "full_map_layers": [int(x) for x in full_map_layers],
        "full_map_heads": [int(x) for x in full_map_heads],
        "full_map_samples": [int(x) for x in sorted(full_map_sample_set)],
        "full_map_specs": [spec.to_dict() for spec in full_map_specs],
        "key_group_order": list(KEY_GROUP_ORDER),
        "query_group_order": list(QUERY_GROUP_ORDER),
        "sample_count": int(len(samples)),
        "samples": samples,
        "matrix_manifest": matrix_manifest,
        "full_map_manifest": full_map_manifest,
        "run_seconds": float(time.perf_counter() - run_start),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "run_complete",
                "sample_count": len(samples),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
