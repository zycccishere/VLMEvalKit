#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from qwen25vl_image2_probe import (  # noqa: E402
    ALL_ATTENTION_FUNCTIONS,
    apply_multimodal_rotary_pos_emb,
    build_base_content,
    build_inputs,
    build_replayed_content,
    eager_attention_forward,
    find_image_spans,
    load_model_and_processor,
    parse_attention_layers,
    repeat_kv,
    resolve_input_device,
    sanitize_single_process_env,
    set_seed,
    tensor_to_device,
)
from vlmeval.cross_image_flow_v2 import (  # noqa: E402
    bbox_to_token_indices,
    extract_image_grid_meta,
    normalize_rows,
    safe_float16,
    token_rows_and_cols,
)
from vlmeval.dataset import build_dataset  # noqa: E402
from vlmeval.vlm.qwen2_vl.replay_prompt_template import (  # noqa: E402
    PROMPT_TEMPLATE_DIRECTLY_ANSWER,
    PROMPT_TEMPLATE_IDENTITY,
)
from vlmeval.vlm.replay_image_transform import (  # noqa: E402
    QWEN_DEFAULT_MAX_PIXELS,
    QWEN_DEFAULT_MIN_PIXELS,
    apply_image_transform_to_content,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs shifted image2 prefill flow on Qwen2.5-VL."
    )
    parser.add_argument("--model-path", default="/user/zyc1781/models/Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-ids", nargs="*", default=[])
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--mode", default="image_text_image")
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=[
            "shift_right_half_vit_token",
            "shift_right_one_vit_token",
            "shift_right_one_llm_token",
        ],
        help="Image2 transforms to compare against baseline. Baseline is always included once.",
    )
    parser.add_argument(
        "--policy",
        default=PROMPT_TEMPLATE_IDENTITY,
        choices=[PROMPT_TEMPLATE_IDENTITY, PROMPT_TEMPLATE_DIRECTLY_ANSWER],
    )
    parser.add_argument("--template-on-last-replay-text", action="store_true")
    parser.add_argument("--attn-layers", default="last4")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--band-radius", type=int, default=1)
    parser.add_argument("--qwen-min-pixels", type=int, default=QWEN_DEFAULT_MIN_PIXELS)
    parser.add_argument("--qwen-max-pixels", type=int, default=QWEN_DEFAULT_MAX_PIXELS)
    return parser


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Transform-pair manifest must be a JSON list.")
    return payload


def resolve_case_image_path(manifest_path: str | Path, image_ref: str) -> Path:
    image_path = Path(str(image_ref))
    if image_path.is_absolute() and image_path.exists():
        return image_path.resolve()
    if not image_path.is_absolute():
        candidate = (Path(manifest_path).resolve().parent / image_path).resolve()
        if candidate.exists():
            return candidate
        if image_path.exists():
            return image_path.resolve()
    raise FileNotFoundError(f"Case image not found: {image_ref}")


def resolve_dataset_row(dataset, dataset_name: str, sample_index: int) -> pd.Series:
    matches = dataset.data[dataset.data["index"] == sample_index]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row for {dataset_name} index={sample_index}, got {len(matches)}.")
    return matches.iloc[0]


def build_chebyshev_distance(query_rows: np.ndarray, query_cols: np.ndarray, key_rows: np.ndarray, key_cols: np.ndarray) -> np.ndarray:
    return np.maximum(np.abs(query_rows[:, None] - key_rows[None, :]), np.abs(query_cols[:, None] - key_cols[None, :]))


def build_euclidean_distance(query_rows: np.ndarray, query_cols: np.ndarray, key_rows: np.ndarray, key_cols: np.ndarray) -> np.ndarray:
    return np.sqrt((query_rows[:, None] - key_rows[None, :]) ** 2 + (query_cols[:, None] - key_cols[None, :]) ** 2)


def local_correspondence_band_mass(matrix_norm: np.ndarray, cheb_dist: np.ndarray, radius: int) -> float:
    mask = cheb_dist <= radius
    return float(matrix_norm[mask].sum() / max(matrix_norm.shape[0], 1))


def expected_distance_from_diagonal(matrix_norm: np.ndarray, euclid_dist: np.ndarray) -> float:
    max_dist = float(np.max(euclid_dist)) if euclid_dist.size else 0.0
    if max_dist <= 0:
        return 0.0
    norm_dist = euclid_dist / max_dist
    return float((matrix_norm * norm_dist).sum(axis=-1).mean())


def row_entropy(matrix_norm: np.ndarray) -> float:
    if matrix_norm.size == 0:
        return float("nan")
    eps = 1e-8
    row_logs = np.log(np.clip(matrix_norm, eps, 1.0))
    entropy = -(matrix_norm * row_logs).sum(axis=-1)
    denom = math.log(matrix_norm.shape[1]) if matrix_norm.shape[1] > 1 else 1.0
    return float((entropy / denom).mean())


def distance_profile(matrix_norm: np.ndarray, cheb_dist: np.ndarray) -> list[float]:
    if matrix_norm.size == 0:
        return []
    max_d = int(np.max(cheb_dist)) if cheb_dist.size else 0
    values: list[float] = []
    for dist in range(max_d + 1):
        mask = cheb_dist == dist
        values.append(float(matrix_norm[mask].sum() / max(matrix_norm.shape[0], 1)))
    return values


def finite_or_nan(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def image_size_from_content(content: list[dict[str, Any]]) -> tuple[int, int] | None:
    for item in content:
        if item.get("type") != "image":
            continue
        ref = str(item.get("image") or item.get("image_url") or "").strip()
        if ref.startswith("file://"):
            ref = ref[len("file://") :]
        if not ref:
            continue
        try:
            with Image.open(ref) as image:
                return tuple(int(v) for v in image.size)
        except Exception:
            return None
    return None


def attach_qwen_pixel_budget(
    content: list[dict[str, Any]],
    *,
    min_pixels: int,
    max_pixels: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in content:
        copied = dict(item)
        if copied.get("type") == "image":
            if min_pixels > 0:
                copied["min_pixels"] = int(min_pixels)
            if max_pixels > 0:
                copied["max_pixels"] = int(max_pixels)
        out.append(copied)
    return out


def explicit_content_for_case(manifest_path: str | Path, case: dict[str, Any]) -> list[dict[str, Any]]:
    image_path = resolve_case_image_path(manifest_path, str(case["image"]))
    question = str(case["question"])
    return [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": question},
    ]


def case_dataset_name(case: dict[str, Any]) -> str:
    if case.get("source_dataset"):
        return str(case["source_dataset"])
    source = case.get("source") if isinstance(case.get("source"), dict) else {}
    return str(source.get("dataset") or "controlled_target_box")


def case_sample_index(case: dict[str, Any], fallback: int) -> int | str:
    for key in ("source_index", "sample_index", "selection_rank"):
        if key in case:
            return case[key]
    source = case.get("source") if isinstance(case.get("source"), dict) else {}
    for key in ("row_index", "source_index"):
        if key in source:
            return source[key]
    return fallback


def target_mass_summary(
    *,
    matrix_norm: np.ndarray,
    image_size: tuple[int, int] | None,
    key_grid_meta,
    query_grid_meta,
    transform_record: dict[str, Any],
    target_box_xyxy: list[int] | None,
    distractor_box_xyxy: list[int] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "target_key_token_indices": [],
        "target_query_token_indices": [],
        "content_shifted_target_query_token_indices": [],
        "distractor_key_token_indices": [],
        "target_mass_norm_all_queries": float("nan"),
        "target_mass_norm_target_queries": float("nan"),
        "target_mass_norm_content_shifted_target_queries": float("nan"),
        "distractor_mass_norm_all_queries": float("nan"),
        "target_minus_distractor_mass": float("nan"),
    }
    if image_size is None or not target_box_xyxy:
        return out
    target_indices = bbox_to_token_indices(
        image_size=image_size,
        grid_meta=key_grid_meta,
        bbox_xyxy=[int(x) for x in target_box_xyxy],
    )
    target_query_indices = bbox_to_token_indices(
        image_size=image_size,
        grid_meta=query_grid_meta,
        bbox_xyxy=[int(x) for x in target_box_xyxy],
    )
    shifted_target_query_indices = content_shifted_bbox_token_indices(
        image_size=image_size,
        grid_meta=query_grid_meta,
        bbox_xyxy=[int(x) for x in target_box_xyxy],
        transform_record=transform_record,
    )
    out["target_key_token_indices"] = [int(x) for x in target_indices]
    out["target_query_token_indices"] = [int(x) for x in target_query_indices]
    out["content_shifted_target_query_token_indices"] = [int(x) for x in shifted_target_query_indices]
    if target_indices:
        target_mass = float(matrix_norm[:, target_indices].sum(axis=-1).mean())
        out["target_mass_norm_all_queries"] = target_mass
        if target_query_indices:
            out["target_mass_norm_target_queries"] = float(
                matrix_norm[target_query_indices][:, target_indices].sum(axis=-1).mean()
            )
        if shifted_target_query_indices:
            out["target_mass_norm_content_shifted_target_queries"] = float(
                matrix_norm[shifted_target_query_indices][:, target_indices].sum(axis=-1).mean()
            )
    if distractor_box_xyxy:
        distractor_indices = bbox_to_token_indices(
            image_size=image_size,
            grid_meta=key_grid_meta,
            bbox_xyxy=[int(x) for x in distractor_box_xyxy],
        )
        out["distractor_key_token_indices"] = [int(x) for x in distractor_indices]
        if distractor_indices:
            distractor_mass = float(matrix_norm[:, distractor_indices].sum(axis=-1).mean())
            out["distractor_mass_norm_all_queries"] = distractor_mass
            if target_indices:
                out["target_minus_distractor_mass"] = float(out["target_mass_norm_all_queries"] - distractor_mass)
    return out


def content_shifted_bbox_token_indices(
    *,
    image_size: tuple[int, int],
    grid_meta,
    bbox_xyxy: list[int],
    transform_record: dict[str, Any],
) -> list[int]:
    width, height = image_size
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    shift = transform_record.get("shift") or {}
    processed_width = float(shift.get("processed_resized_width") or width)
    processed_height = float(shift.get("processed_resized_height") or height)
    dx = float(shift.get("dx", 0.0))
    dy = float(shift.get("dy", 0.0))
    if processed_width <= 0 or processed_height <= 0:
        return []
    token_w = processed_width / max(int(grid_meta.llm_grid_w), 1)
    token_h = processed_height / max(int(grid_meta.llm_grid_h), 1)
    indices: list[int] = []
    rows, cols = token_rows_and_cols(grid_meta)
    for token_index, (row, col) in enumerate(zip(rows.tolist(), cols.tolist())):
        source_x_proc = ((float(col) + 0.5) * token_w - dx) % processed_width
        source_y_proc = ((float(row) + 0.5) * token_h - dy) % processed_height
        source_x = source_x_proc / processed_width * float(width)
        source_y = source_y_proc / processed_height * float(height)
        if x1 <= source_x <= x2 and y1 <= source_y <= y2:
            indices.append(int(token_index))
    return indices


def wrapped_distance(values: np.ndarray, centers: np.ndarray, period: int) -> np.ndarray:
    direct = np.abs(values[None, :] - centers[:, None])
    if period <= 0:
        return direct
    return np.minimum(direct, period - direct)


def content_correspondence_distances(
    *,
    query_rows: np.ndarray,
    query_cols: np.ndarray,
    key_rows: np.ndarray,
    key_cols: np.ndarray,
    grid_h: int,
    grid_w: int,
    transform_record: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    shift = transform_record.get("shift") or {}
    stride = float(shift.get("llm_visual_token_stride") or shift.get("qwen_token_stride") or 28.0)
    dx_tokens = float(shift.get("dx", 0.0)) / stride
    dy_tokens = float(shift.get("dy", 0.0)) / stride
    source_cols = (query_cols.astype(np.float64) - dx_tokens) % max(grid_w, 1)
    source_rows = (query_rows.astype(np.float64) - dy_tokens) % max(grid_h, 1)
    dc = wrapped_distance(key_cols.astype(np.float64), source_cols, grid_w)
    dr = wrapped_distance(key_rows.astype(np.float64), source_rows, grid_h)
    cheb = np.maximum(dr, dc)
    euclid = np.sqrt(dr**2 + dc**2)
    meta = {
        "dx_tokens": dx_tokens,
        "dy_tokens": dy_tokens,
        "llm_visual_token_stride": stride,
    }
    return cheb, euclid, meta


def i2_self_flow_summary(image2_block: np.ndarray, query_rows: np.ndarray, query_cols: np.ndarray, radius: int) -> dict[str, float]:
    if image2_block.size == 0:
        return {
            "i2_total_self_mass_raw": float("nan"),
            "i2_past_self_mass_raw": float("nan"),
            "i2_diag_self_mass_raw": float("nan"),
            "i2_local_self_mass_raw": float("nan"),
            "i2_local_self_ratio": float("nan"),
        }
    n = int(image2_block.shape[0])
    if image2_block.shape[1] != n:
        return {
            "i2_total_self_mass_raw": float("nan"),
            "i2_past_self_mass_raw": float("nan"),
            "i2_diag_self_mass_raw": float("nan"),
            "i2_local_self_mass_raw": float("nan"),
            "i2_local_self_ratio": float("nan"),
        }
    idx = np.arange(n)
    past_mask = idx[None, :] < idx[:, None]
    diag_mask = idx[None, :] == idx[:, None]
    cheb = build_chebyshev_distance(query_rows, query_cols, query_rows, query_cols)
    local_mask = past_mask & (cheb <= radius)
    total_mass = float(image2_block.sum(axis=-1).mean())
    past_mass = float(image2_block[past_mask].sum() / max(n, 1))
    diag_mass = float(image2_block[diag_mask].sum() / max(n, 1))
    local_mass = float(image2_block[local_mask].sum() / max(n, 1))
    return {
        "i2_total_self_mass_raw": total_mass,
        "i2_past_self_mass_raw": past_mass,
        "i2_diag_self_mass_raw": diag_mass,
        "i2_local_self_mass_raw": local_mass,
        "i2_local_self_ratio": float(local_mass / max(past_mass, 1e-8)),
    }


def unique_transforms(case: dict[str, Any], requested: list[str]) -> list[str]:
    transforms: list[str] = ["baseline"]
    for item in requested:
        if item and item not in transforms:
            transforms.append(str(item))
    case_transform = str(case.get("shift_transform", "")).strip()
    if case_transform and case_transform not in transforms:
        transforms.append(case_transform)
    return transforms


class QwenTransformPairFlowTracer:
    def __init__(self, attn_modules: dict[int, Any]):
        self.attn_modules = attn_modules
        self.original_forward: dict[int, Any] = {}
        self.records: list[dict[str, Any]] = []
        self.query_positions: list[int] = []
        self.image1_positions: list[int] = []
        self.text_positions: list[int] = []
        self.image2_positions: list[int] = []

    def configure_sample(
        self,
        *,
        query_positions: list[int],
        image1_positions: list[int],
        text_positions: list[int],
        image2_positions: list[int],
    ) -> None:
        self.query_positions = list(query_positions)
        self.image1_positions = list(image1_positions)
        self.text_positions = list(text_positions)
        self.image2_positions = list(image2_positions)

    def patch(self) -> None:
        tracer = self
        for layer_idx, attn_module in self.attn_modules.items():
            self.original_forward[layer_idx] = attn_module.forward

            def instrumented_forward(
                module,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                position_ids: torch.LongTensor | None = None,
                past_key_value=None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: torch.LongTensor | None = None,
                position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
                _layer_idx: int = layer_idx,
                **kwargs,
            ):
                bsz, q_len, _ = hidden_states.size()

                query_states = module.q_proj(hidden_states)
                key_states = module.k_proj(hidden_states)
                value_states = module.v_proj(hidden_states)

                query_states = query_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)

                if position_embeddings is None:
                    raise ValueError("position_embeddings is required for transform-pair flow tracing.")
                cos, sin = position_embeddings
                query_states, key_states = apply_multimodal_rotary_pos_emb(
                    query_states,
                    key_states,
                    cos,
                    sin,
                    module.rope_scaling["mrope_section"],
                )

                if past_key_value is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_value.update(
                        key_states,
                        value_states,
                        module.layer_idx,
                        cache_kwargs,
                    )

                key_states_for_scores = repeat_kv(key_states, module.num_key_value_groups)

                if q_len > 1 and tracer.query_positions:
                    query_index_tensor = torch.as_tensor(
                        tracer.query_positions,
                        device=query_states.device,
                        dtype=torch.long,
                    )
                    query_slice = torch.index_select(query_states, dim=2, index=query_index_tensor)
                    scores = torch.matmul(query_slice, key_states_for_scores.transpose(2, 3)) * module.scaling
                    if attention_mask is not None:
                        selected_mask = attention_mask[:, :, tracer.query_positions, : key_states_for_scores.shape[-2]]
                        scores = scores + selected_mask
                    key_position_tensor = torch.arange(
                        key_states_for_scores.shape[-2],
                        device=query_states.device,
                        dtype=torch.long,
                    )
                    causal_block = key_position_tensor.view(1, 1, 1, -1) > query_index_tensor.view(1, 1, -1, 1)
                    scores = scores.masked_fill(causal_block, torch.finfo(scores.dtype).min)
                    attn_sel = F.softmax(scores, dim=-1, dtype=torch.float32)
                    mean_attn = attn_sel.mean(dim=1).squeeze(0)
                    tracer.records.append(
                        {
                            "layer": int(_layer_idx),
                            "image1_block": mean_attn[:, tracer.image1_positions].detach().cpu().numpy(),
                            "image2_block": (
                                mean_attn[:, tracer.image2_positions].detach().cpu().numpy()
                                if tracer.image2_positions
                                else np.zeros((len(tracer.query_positions), 0), dtype=np.float32)
                            ),
                            "image1_mass_raw": mean_attn[:, tracer.image1_positions].sum(dim=-1).detach().cpu().numpy(),
                            "text_mass_raw": (
                                mean_attn[:, tracer.text_positions].sum(dim=-1).detach().cpu().numpy()
                                if tracer.text_positions
                                else np.zeros((len(tracer.query_positions),), dtype=np.float32)
                            ),
                            "image2_mass_raw": (
                                mean_attn[:, tracer.image2_positions].sum(dim=-1).detach().cpu().numpy()
                                if tracer.image2_positions
                                else np.zeros((len(tracer.query_positions),), dtype=np.float32)
                            ),
                        }
                    )

                attention_interface = eager_attention_forward
                if module.config._attn_implementation != "eager":
                    attention_interface = ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation]

                attn_output, attn_weights = attention_interface(
                    module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not module.training else module.attention_dropout,
                    scaling=module.scaling,
                    sliding_window=module.sliding_window,
                    **kwargs,
                )

                attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
                attn_output = module.o_proj(attn_output)
                if not output_attentions:
                    attn_weights = None
                return attn_output, attn_weights

            attn_module.forward = types.MethodType(instrumented_forward, attn_module)

    def restore(self) -> None:
        for layer_idx, attn_module in self.attn_modules.items():
            attn_module.forward = self.original_forward[layer_idx]

    def reset(self) -> None:
        self.records.clear()


def main() -> int:
    args = build_parser().parse_args()
    sanitize_single_process_env()
    set_seed(args.seed)

    manifest = load_manifest(args.manifest)
    if args.case_ids:
        keep = set(args.case_ids)
        manifest = [item for item in manifest if item["id"] in keep]
    if args.max_cases > 0:
        manifest = manifest[: args.max_cases]
    if not manifest:
        raise ValueError("No cases selected from manifest.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = output_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    processor, model = load_model_and_processor(args.model_path, args.device)
    input_device = resolve_input_device(model, args.device)
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    special_token_ids = set(processor.tokenizer.all_special_ids)
    spatial_merge_size = int(model.model.visual.spatial_merge_size)

    selected_layers = parse_attention_layers(args.attn_layers, len(model.model.language_model.layers))
    tracer = QwenTransformPairFlowTracer(
        {layer_idx: model.model.language_model.layers[layer_idx].self_attn for layer_idx in selected_layers}
    )
    tracer.patch()

    records_out: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    dataset_cache: dict[str, Any] = {}

    try:
        for case_idx, case in enumerate(manifest):
            dataset_name = case_dataset_name(case)
            sample_index = case_sample_index(case, case_idx)
            if case.get("image") and case.get("question"):
                base_content_raw = explicit_content_for_case(args.manifest, case)
            else:
                dataset = dataset_cache.get(dataset_name)
                if dataset is None:
                    dataset = build_dataset(dataset_name)
                    dataset_cache[dataset_name] = dataset
                row = resolve_dataset_row(dataset, dataset_name, int(sample_index))
                base_content_raw = build_base_content(dataset, row)
            base_content = attach_qwen_pixel_budget(
                base_content_raw,
                min_pixels=args.qwen_min_pixels,
                max_pixels=args.qwen_max_pixels,
            )
            base_image_size = image_size_from_content(base_content)

            transform_records: dict[str, Any] = {}
            layer_results: dict[str, list[dict[str, Any]]] = {}
            transforms_for_case = unique_transforms(case, args.transforms)
            for transform in transforms_for_case:
                sample_start = time.perf_counter()
                replayed = build_replayed_content(
                    base_content,
                    dataset_name=dataset_name,
                    mode=args.mode,
                    policy=args.policy,
                    template_on_last_replay_text=args.template_on_last_replay_text,
                )
                transformed, transform_record = apply_image_transform_to_content(
                    replayed,
                    transform_name=transform,
                    sample_meta={"sample_index": sample_index},
                    cache_dir=output_dir / "_transform_cache" / transform / dataset_name,
                    dataset_name=dataset_name,
                    image_position=2,
                    model_family="qwen2_5_vl",
                )
                _, prompt_text, model_inputs = build_inputs(processor, transformed)
                input_ids = model_inputs["input_ids"][0].tolist()
                image_spans = find_image_spans(input_ids, image_token_id)
                if len(image_spans) != 2:
                    raise ValueError(
                        f"Expected exactly 2 image spans for {case['id']} transform={transform}, got {len(image_spans)}."
                    )
                image1_positions = list(range(image_spans[0].start, image_spans[0].end + 1))
                image2_positions = list(range(image_spans[1].start, image_spans[1].end + 1))
                text_positions = [
                    pos
                    for pos, token_id in enumerate(input_ids)
                    if pos not in set(image1_positions + image2_positions) and token_id not in special_token_ids
                ]

                grid_metas = extract_image_grid_meta(model_inputs["image_grid_thw"], spatial_merge_size=spatial_merge_size)
                if len(grid_metas) != 2:
                    raise ValueError(
                        f"Expected exactly 2 image grids for {case['id']} transform={transform}, got {len(grid_metas)}."
                    )
                if grid_metas[0].token_count != len(image1_positions) or grid_metas[1].token_count != len(image2_positions):
                    raise ValueError(
                        f"Image-token mismatch for {case['id']} transform={transform}: "
                        f"spans=({len(image1_positions)}, {len(image2_positions)}) grid=({grid_metas[0].token_count}, {grid_metas[1].token_count})"
                    )

                query_rows, query_cols = token_rows_and_cols(grid_metas[1])
                key_rows, key_cols = token_rows_and_cols(grid_metas[0])
                cheb_dist = build_chebyshev_distance(query_rows, query_cols, key_rows, key_cols)
                euclid_dist = build_euclidean_distance(query_rows, query_cols, key_rows, key_cols)
                content_cheb_dist, content_euclid_dist, content_shift_meta = content_correspondence_distances(
                    query_rows=query_rows,
                    query_cols=query_cols,
                    key_rows=key_rows,
                    key_cols=key_cols,
                    grid_h=grid_metas[0].llm_grid_h,
                    grid_w=grid_metas[0].llm_grid_w,
                    transform_record=transform_record,
                )

                tracer.configure_sample(
                    query_positions=image2_positions,
                    image1_positions=image1_positions,
                    text_positions=text_positions,
                    image2_positions=image2_positions,
                )
                tracer.reset()

                model_inputs = tensor_to_device(model_inputs, input_device)
                with torch.inference_mode():
                    outputs = model(**model_inputs, use_cache=False, return_dict=True)
                del outputs
                torch.cuda.empty_cache()

                if not tracer.records:
                    raise RuntimeError(f"No flow records captured for {case['id']} transform={transform}.")

                transform_records[transform] = {
                    "prompt_text": prompt_text,
                    "transform_record": transform_record,
                    "content_shift_meta": content_shift_meta,
                    "seconds": float(time.perf_counter() - sample_start),
                    "image1_grid": grid_metas[0].to_dict(),
                    "image2_grid": grid_metas[1].to_dict(),
                }

                layer_summaries: list[dict[str, Any]] = []
                for record in tracer.records:
                    layer = int(record["layer"])
                    raw_block = np.asarray(record["image1_block"], dtype=np.float32)
                    image2_block = np.asarray(record["image2_block"], dtype=np.float32)
                    norm_block = normalize_rows(raw_block)
                    image1_mass_raw = np.asarray(record["image1_mass_raw"], dtype=np.float32)
                    text_mass_raw = np.asarray(record["text_mass_raw"], dtype=np.float32)
                    image2_mass_raw = np.asarray(record["image2_mass_raw"], dtype=np.float32)
                    profile = distance_profile(norm_block, cheb_dist)
                    content_profile = distance_profile(norm_block, content_cheb_dist)
                    self_summary = i2_self_flow_summary(image2_block, query_rows, query_cols, args.band_radius)
                    target_summary = target_mass_summary(
                        matrix_norm=norm_block,
                        image_size=base_image_size,
                        key_grid_meta=grid_metas[0],
                        query_grid_meta=grid_metas[1],
                        transform_record=transform_record,
                        target_box_xyxy=case.get("target_box_xyxy"),
                        distractor_box_xyxy=case.get("distractor_box_xyxy"),
                    )
                    npz_rel = Path("npz") / f"{case['id']}__{transform}__layer{layer}.npz"
                    np.savez_compressed(
                        output_dir / npz_rel,
                        matrix_raw=safe_float16(raw_block),
                        matrix_norm=safe_float16(norm_block),
                        image2_block_raw=safe_float16(image2_block),
                        image1_mass_raw=safe_float16(image1_mass_raw),
                        text_mass_raw=safe_float16(text_mass_raw),
                        image2_mass_raw=safe_float16(image2_mass_raw),
                        query_rows=query_rows.astype(np.int16),
                        query_cols=query_cols.astype(np.int16),
                        key_rows=key_rows.astype(np.int16),
                        key_cols=key_cols.astype(np.int16),
                    )
                    layer_summaries.append(
                        {
                            "layer": layer,
                            "npz_path": str(npz_rel),
                            "position_band_mass": local_correspondence_band_mass(norm_block, cheb_dist, args.band_radius),
                            "local_correspondence_band_mass": local_correspondence_band_mass(norm_block, cheb_dist, args.band_radius),
                            "content_band_mass": local_correspondence_band_mass(norm_block, content_cheb_dist, args.band_radius),
                            "expected_position_distance": expected_distance_from_diagonal(norm_block, euclid_dist),
                            "expected_content_distance": expected_distance_from_diagonal(norm_block, content_euclid_dist),
                            "expected_distance_from_diagonal": expected_distance_from_diagonal(norm_block, euclid_dist),
                            "row_entropy": row_entropy(norm_block),
                            "mean_image1_mass_raw": float(np.asarray(image1_mass_raw, dtype=np.float64).mean()),
                            "mean_text_mass_raw": float(np.asarray(text_mass_raw, dtype=np.float64).mean()),
                            "mean_image2_mass_raw": float(np.asarray(image2_mass_raw, dtype=np.float64).mean()),
                            **self_summary,
                            **target_summary,
                            "distance_profile": profile,
                            "content_distance_profile": content_profile,
                        }
                    )
                layer_results[transform] = sorted(layer_summaries, key=lambda item: item["layer"])
                tracer.reset()

            record = {
                "case_id": case["id"],
                "base_id": case.get("base_id", ""),
                "question_id": case.get("question_id", ""),
                "group": case.get("group", ""),
                "source_dataset": dataset_name,
                "source_index": sample_index,
                "question": str(case.get("question", "")),
                "answer": str(case.get("answer", "")),
                "image": str(case.get("image", "")),
                "image_size": list(base_image_size) if base_image_size else None,
                "target_box_xyxy": case.get("target_box_xyxy"),
                "distractor_box_xyxy": case.get("distractor_box_xyxy"),
                "mode": args.mode,
                "policy": args.policy,
                "selected_layers": selected_layers,
                "transforms": {
                    transform: {**transform_records[transform], "layers": layer_results[transform]}
                    for transform in transforms_for_case
                },
            }
            records_out.append(record)
            print(
                json.dumps(
                    {
                            "event": "case_complete",
                            "case_id": case["id"],
                            "transforms": transforms_for_case,
                        },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        tracer.restore()

    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "model_path": args.model_path,
        "mode": args.mode,
        "policy": args.policy,
        "attn_layers": args.attn_layers,
        "selected_layers": selected_layers,
        "transforms": args.transforms,
        "band_radius": args.band_radius,
        "qwen_min_pixels": args.qwen_min_pixels,
        "qwen_max_pixels": args.qwen_max_pixels,
        "case_count": len(records_out),
        "run_seconds": float(time.perf_counter() - run_start),
        "cases": records_out,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "run_complete", "case_count": len(records_out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
