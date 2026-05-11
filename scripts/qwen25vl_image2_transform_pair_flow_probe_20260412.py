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
    apply_image_transform_to_content,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs shifted image2 cross-image flow on Qwen2.5-VL-32B."
    )
    parser.add_argument("--model-path", default="/models/Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-ids", nargs="*", default=[])
    parser.add_argument("--mode", default="image_text_image")
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
    return parser


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Transform-pair manifest must be a JSON list.")
    return payload


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
                    attn_sel = F.softmax(scores, dim=-1, dtype=torch.float32)
                    mean_attn = attn_sel.mean(dim=1).squeeze(0)
                    tracer.records.append(
                        {
                            "layer": int(_layer_idx),
                            "image1_block": mean_attn[:, tracer.image1_positions].detach().cpu().numpy(),
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
                return attn_output, attn_weights, past_key_value

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
        for case in manifest:
            dataset_name = str(case["source_dataset"])
            sample_index = int(case["source_index"])
            shift_transform = str(case["shift_transform"])
            dataset = dataset_cache.get(dataset_name)
            if dataset is None:
                dataset = build_dataset(dataset_name)
                dataset_cache[dataset_name] = dataset
            row = resolve_dataset_row(dataset, dataset_name, sample_index)
            base_content = build_base_content(dataset, row)

            transform_records: dict[str, Any] = {}
            layer_results: dict[str, list[dict[str, Any]]] = {}
            for transform in ["baseline", shift_transform]:
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
                    "seconds": float(time.perf_counter() - sample_start),
                    "image1_grid": grid_metas[0].to_dict(),
                    "image2_grid": grid_metas[1].to_dict(),
                }

                layer_summaries: list[dict[str, Any]] = []
                for record in tracer.records:
                    layer = int(record["layer"])
                    raw_block = np.asarray(record["image1_block"], dtype=np.float32)
                    norm_block = normalize_rows(raw_block)
                    image1_mass_raw = np.asarray(record["image1_mass_raw"], dtype=np.float32)
                    text_mass_raw = np.asarray(record["text_mass_raw"], dtype=np.float32)
                    image2_mass_raw = np.asarray(record["image2_mass_raw"], dtype=np.float32)
                    profile = distance_profile(norm_block, cheb_dist)
                    npz_rel = Path("npz") / f"{case['id']}__{transform}__layer{layer}.npz"
                    np.savez_compressed(
                        output_dir / npz_rel,
                        matrix_raw=safe_float16(raw_block),
                        matrix_norm=safe_float16(norm_block),
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
                            "local_correspondence_band_mass": local_correspondence_band_mass(norm_block, cheb_dist, args.band_radius),
                            "expected_distance_from_diagonal": expected_distance_from_diagonal(norm_block, euclid_dist),
                            "row_entropy": row_entropy(norm_block),
                            "mean_image1_mass_raw": float(np.asarray(image1_mass_raw, dtype=np.float64).mean()),
                            "mean_text_mass_raw": float(np.asarray(text_mass_raw, dtype=np.float64).mean()),
                            "mean_image2_mass_raw": float(np.asarray(image2_mass_raw, dtype=np.float64).mean()),
                            "distance_profile": profile,
                        }
                    )
                layer_results[transform] = sorted(layer_summaries, key=lambda item: item["layer"])
                tracer.reset()

            record = {
                "case_id": case["id"],
                "group": case.get("group", ""),
                "source_dataset": dataset_name,
                "source_index": sample_index,
                "question": str(case.get("question", "")),
                "answer": str(case.get("answer", "")),
                "mode": args.mode,
                "policy": args.policy,
                "selected_layers": selected_layers,
                "transforms": {
                    "baseline": {**transform_records["baseline"], "layers": layer_results["baseline"]},
                    shift_transform: {**transform_records[shift_transform], "layers": layer_results[shift_transform]},
                },
            }
            records_out.append(record)
            print(
                json.dumps(
                    {
                        "event": "case_complete",
                        "case_id": case["id"],
                        "shift_transform": shift_transform,
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
        "band_radius": args.band_radius,
        "case_count": len(records_out),
        "run_seconds": float(time.perf_counter() - run_start),
        "cases": records_out,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "run_complete", "case_count": len(records_out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
