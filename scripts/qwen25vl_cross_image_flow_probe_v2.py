#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from qwen25vl_image2_probe import (  # noqa: E402
    ALL_ATTENTION_FUNCTIONS,
    apply_multimodal_rotary_pos_emb,
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
    flatten_controlled_manifest,
    image_token_table,
    load_controlled_manifest,
    normalize_rows,
    resolve_manifest_image_path,
    safe_float16,
)
from vlmeval.vlm.qwen2_vl.replay_prompt_template import (  # noqa: E402
    PROMPT_TEMPLATE_DIRECTLY_ANSWER,
    PROMPT_TEMPLATE_IDENTITY,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Patch-level cross-image flow probe for Qwen2.5-VL controlled IIT vs ITI analysis."
    )
    parser.add_argument("--model-path", default="/models/Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-ids", nargs="*", default=[])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["image_image_text", "image_text_image"],
        help="Replay modes to compare. Defaults to IIT and ITI.",
    )
    parser.add_argument(
        "--policy",
        default=PROMPT_TEMPLATE_IDENTITY,
        choices=[PROMPT_TEMPLATE_IDENTITY, PROMPT_TEMPLATE_DIRECTLY_ANSWER],
    )
    parser.add_argument("--template-on-last-replay-text", action="store_true")
    parser.add_argument(
        "--attn-layers",
        default="last4",
        help="Attention layers to export: last, all, last4, or comma-separated indices.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def base_content_for_case(image_path: Path, question: str) -> list[dict[str, Any]]:
    return [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": question},
    ]


def mode_alias(mode: str) -> str:
    if mode == "image_text_image":
        return "ITI"
    if mode == "image_image_text":
        return "IIT"
    return mode


def find_text_positions_all(input_ids: list[int], image_spans: list[Any], special_token_ids: set[int]) -> list[int]:
    image_positions: set[int] = set()
    for span in image_spans:
        image_positions.update(range(int(span.start), int(span.end) + 1))
    text_positions: list[int] = []
    for pos, token_id in enumerate(input_ids):
        if pos in image_positions:
            continue
        if token_id in special_token_ids:
            continue
        text_positions.append(int(pos))
    return text_positions


def build_neighborhood_mask(rows: np.ndarray, cols: np.ndarray, radius: int = 1) -> np.ndarray:
    if len(rows) != len(cols):
        raise ValueError("rows and cols length mismatch")
    dr = np.abs(rows[:, None] - rows[None, :])
    dc = np.abs(cols[:, None] - cols[None, :])
    return (dr <= radius) & (dc <= radius)


def mean_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.asarray(values, dtype=np.float64).mean())


def masked_row_mean(matrix: np.ndarray, mask_indices: list[int]) -> float:
    if matrix.size == 0 or not mask_indices:
        return float("nan")
    return float(np.asarray(matrix[:, mask_indices], dtype=np.float64).sum(axis=-1).mean())


def mean_diag_mass(matrix_norm: np.ndarray, neighborhood_mask: np.ndarray) -> float:
    if matrix_norm.size == 0:
        return float("nan")
    values = matrix_norm[neighborhood_mask]
    q_count = matrix_norm.shape[0]
    return float(values.sum() / max(q_count, 1))


def derive_layer_summary(
    *,
    matrix_norm: np.ndarray,
    image1_mass_raw: np.ndarray,
    text_mass_raw: np.ndarray,
    image2_mass_raw: np.ndarray,
    target_key_indices: list[int],
    target_query_indices: list[int],
    neighborhood_mask: np.ndarray,
) -> dict[str, float]:
    query_mask = np.zeros(matrix_norm.shape[0], dtype=bool)
    if target_query_indices:
        query_mask[target_query_indices] = True
    other_query_mask = ~query_mask
    target_mass_all = masked_row_mean(matrix_norm, target_key_indices)
    target_mass_target_queries = (
        float(matrix_norm[query_mask][:, target_key_indices].sum(axis=-1).mean())
        if target_query_indices and query_mask.any()
        else float("nan")
    )
    target_mass_other_queries = (
        float(matrix_norm[other_query_mask][:, target_key_indices].sum(axis=-1).mean())
        if target_key_indices and other_query_mask.any()
        else float("nan")
    )
    diag_all = mean_diag_mass(matrix_norm, neighborhood_mask)
    diag_other = (
        float(matrix_norm[other_query_mask][neighborhood_mask[other_query_mask]].sum() / max(int(other_query_mask.sum()), 1))
        if other_query_mask.any()
        else float("nan")
    )
    return {
        "mean_image1_mass_raw": mean_or_nan(image1_mass_raw),
        "mean_text_mass_raw": mean_or_nan(text_mass_raw),
        "mean_image2_mass_raw": mean_or_nan(image2_mass_raw),
        "target_box_mass_norm_all_queries": target_mass_all,
        "target_box_mass_norm_target_queries": target_mass_target_queries,
        "target_box_mass_norm_other_queries": target_mass_other_queries,
        "diag_mass_norm_all_queries": diag_all,
        "diag_mass_norm_other_queries": diag_other,
    }


class QwenCrossImageFlowTracer:
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
                    raise ValueError("position_embeddings is required for cross-image flow tracing.")
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
                            "query_positions": list(tracer.query_positions),
                            "image1_positions": list(tracer.image1_positions),
                            "text_positions": list(tracer.text_positions),
                            "image2_positions": list(tracer.image2_positions),
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


def generate_text_for_case(
    *,
    model,
    processor,
    model_inputs: dict[str, Any],
    max_new_tokens: int,
) -> tuple[list[int], str]:
    with torch.inference_mode():
        sequences = model.generate(
            **model_inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=max_new_tokens,
        )
    prompt_len = model_inputs["input_ids"].shape[-1]
    generated_ids = sequences[0, prompt_len:].detach().cpu().tolist()
    generated_text = processor.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    del sequences
    torch.cuda.empty_cache()
    return generated_ids, generated_text


def main() -> int:
    args = build_parser().parse_args()
    sanitize_single_process_env()
    set_seed(args.seed)

    manifest = load_controlled_manifest(args.manifest)
    cases = flatten_controlled_manifest(manifest)
    if args.case_ids:
        keep = set(args.case_ids)
        cases = [case for case in cases if case["case_id"] in keep or case["base_id"] in keep]
    if not cases:
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
    tracer = QwenCrossImageFlowTracer(
        {layer_idx: model.model.language_model.layers[layer_idx].self_attn for layer_idx in selected_layers}
    )
    tracer.patch()

    records_out: list[dict[str, Any]] = []
    run_start = time.perf_counter()

    try:
        for case in cases:
            image_path = resolve_manifest_image_path(args.manifest, case["image"])
            with Image.open(image_path) as image:
                image_size = tuple(int(v) for v in image.size)
            for mode in args.modes:
                sample_start = time.perf_counter()
                base_content = base_content_for_case(image_path, case["question"])
                content = build_replayed_content(
                    base_content,
                    dataset_name="cross_image_flow_controlled",
                    mode=mode,
                    policy=args.policy,
                    template_on_last_replay_text=args.template_on_last_replay_text,
                )
                _, prompt_text, model_inputs = build_inputs(processor, content)
                input_ids = model_inputs["input_ids"][0].tolist()
                image_spans = find_image_spans(input_ids, image_token_id)
                if len(image_spans) != 2:
                    raise ValueError(
                        f"Expected exactly 2 image spans for {case['case_id']} mode={mode}, got {len(image_spans)}."
                    )
                text_positions = find_text_positions_all(input_ids, image_spans, special_token_ids)

                grid_metas = extract_image_grid_meta(
                    model_inputs["image_grid_thw"],
                    spatial_merge_size=spatial_merge_size,
                )
                if len(grid_metas) != 2:
                    raise ValueError(
                        f"Expected exactly 2 image grids for {case['case_id']} mode={mode}, got {len(grid_metas)}."
                    )
                if grid_metas[0].token_count != image_spans[0].length or grid_metas[1].token_count != image_spans[1].length:
                    raise ValueError(
                        f"Image token count mismatch for {case['case_id']} mode={mode}: "
                        f"spans=({image_spans[0].length}, {image_spans[1].length}) "
                        f"grid=({grid_metas[0].token_count}, {grid_metas[1].token_count})."
                    )

                query_positions = list(range(image_spans[1].start, image_spans[1].end + 1))
                image1_positions = list(range(image_spans[0].start, image_spans[0].end + 1))
                image2_positions = list(range(image_spans[1].start, image_spans[1].end + 1))

                target_key_rel = bbox_to_token_indices(
                    image_size=image_size,
                    grid_meta=grid_metas[0],
                    bbox_xyxy=case["target_box_xyxy"],
                )
                target_query_rel = bbox_to_token_indices(
                    image_size=image_size,
                    grid_meta=grid_metas[1],
                    bbox_xyxy=case["target_box_xyxy"],
                )

                query_rows, query_cols = (
                    np.asarray([entry["row"] for entry in image_token_table(grid_metas[1])], dtype=np.int32),
                    np.asarray([entry["col"] for entry in image_token_table(grid_metas[1])], dtype=np.int32),
                )
                key_rows, key_cols = (
                    np.asarray([entry["row"] for entry in image_token_table(grid_metas[0])], dtype=np.int32),
                    np.asarray([entry["col"] for entry in image_token_table(grid_metas[0])], dtype=np.int32),
                )
                if len(query_rows) != len(key_rows):
                    raise ValueError("Query/key image grids must match for identical-image controlled cases.")
                neighborhood_mask = build_neighborhood_mask(query_rows, key_cols * 0 + query_cols, radius=1)
                # correct same-coordinate neighborhood using key rows/cols
                neighborhood_mask = (np.abs(query_rows[:, None] - key_rows[None, :]) <= 1) & (
                    np.abs(query_cols[:, None] - key_cols[None, :]) <= 1
                )

                tracer.configure_sample(
                    query_positions=query_positions,
                    image1_positions=image1_positions,
                    text_positions=text_positions,
                    image2_positions=image2_positions,
                )
                tracer.reset()

                model_inputs = tensor_to_device(model_inputs, input_device)
                with torch.inference_mode():
                    outputs = model(
                        **model_inputs,
                        use_cache=False,
                        return_dict=True,
                    )
                del outputs
                torch.cuda.empty_cache()

                if not tracer.records:
                    raise RuntimeError(f"No cross-image flow records captured for {case['case_id']} mode={mode}.")
                prefill_records = list(tracer.records)
                tracer.reset()

                generated_ids, generated_text = generate_text_for_case(
                    model=model,
                    processor=processor,
                    model_inputs=model_inputs,
                    max_new_tokens=args.max_new_tokens,
                )

                layers: list[int] = []
                image1_raw_blocks: list[np.ndarray] = []
                image1_norm_blocks: list[np.ndarray] = []
                image1_mass_rows: list[np.ndarray] = []
                text_mass_rows: list[np.ndarray] = []
                image2_mass_rows: list[np.ndarray] = []
                layer_summaries: list[dict[str, Any]] = []
                for record in prefill_records:
                    layer = int(record["layer"])
                    raw_block = np.asarray(record["image1_block"], dtype=np.float32)
                    norm_block = normalize_rows(raw_block)
                    image1_mass_raw = np.asarray(record["image1_mass_raw"], dtype=np.float32)
                    text_mass_raw = np.asarray(record["text_mass_raw"], dtype=np.float32)
                    image2_mass_raw = np.asarray(record["image2_mass_raw"], dtype=np.float32)
                    layers.append(layer)
                    image1_raw_blocks.append(safe_float16(raw_block))
                    image1_norm_blocks.append(safe_float16(norm_block))
                    image1_mass_rows.append(safe_float16(image1_mass_raw))
                    text_mass_rows.append(safe_float16(text_mass_raw))
                    image2_mass_rows.append(safe_float16(image2_mass_raw))
                    summary = derive_layer_summary(
                        matrix_norm=norm_block,
                        image1_mass_raw=image1_mass_raw,
                        text_mass_raw=text_mass_raw,
                        image2_mass_raw=image2_mass_raw,
                        target_key_indices=target_key_rel,
                        target_query_indices=target_query_rel,
                        neighborhood_mask=neighborhood_mask,
                    )
                    summary["layer"] = int(layer)
                    layer_summaries.append(summary)

                layers_arr = np.asarray(layers, dtype=np.int32)
                image1_raw_arr = np.stack(image1_raw_blocks, axis=0)
                image1_norm_arr = np.stack(image1_norm_blocks, axis=0)
                image1_mass_arr = np.stack(image1_mass_rows, axis=0)
                text_mass_arr = np.stack(text_mass_rows, axis=0)
                image2_mass_arr = np.stack(image2_mass_rows, axis=0)
                layer_mean_norm = safe_float16(np.asarray(image1_norm_arr, dtype=np.float32).mean(axis=0))

                rel_npz_path = Path("npz") / f"{case['case_id']}__{mode}.npz"
                npz_path = output_dir / rel_npz_path
                np.savez_compressed(
                    npz_path,
                    layers=layers_arr,
                    image1_raw=image1_raw_arr,
                    image1_norm=image1_norm_arr,
                    image1_mass_raw=image1_mass_arr,
                    text_mass_raw=text_mass_arr,
                    image2_mass_raw=image2_mass_arr,
                    layer_mean_norm=layer_mean_norm,
                )

                case_record = {
                    "case_id": case["case_id"],
                    "base_id": case["base_id"],
                    "question_id": case["question_id"],
                    "mode": mode,
                    "mode_alias": mode_alias(mode),
                    "kind": case["kind"],
                    "question": case["question"],
                    "answer": case["answer"],
                    "image": str(image_path),
                    "image_size": list(image_size),
                    "target_box_xyxy": list(case["target_box_xyxy"]),
                    "source": case["source"],
                    "policy": args.policy,
                    "selected_layers": layers,
                    "image1_grid": grid_metas[0].to_dict(),
                    "image2_grid": grid_metas[1].to_dict(),
                    "image1_token_table": image_token_table(grid_metas[0]),
                    "image2_token_table": image_token_table(grid_metas[1]),
                    "target_key_token_indices": [int(x) for x in target_key_rel],
                    "target_query_token_indices": [int(x) for x in target_query_rel],
                    "prompt_text": prompt_text,
                    "generated_ids": [int(x) for x in generated_ids],
                    "generated_text": generated_text,
                    "npz_path": str(rel_npz_path),
                    "layer_summaries": layer_summaries,
                    "seconds": float(time.perf_counter() - sample_start),
                }
                records_out.append(case_record)
                print(
                    json.dumps(
                        {
                            "event": "case_complete",
                            "case_id": case["case_id"],
                            "mode": mode,
                            "seconds": case_record["seconds"],
                            "generated_text": generated_text,
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
        "modes": args.modes,
        "policy": args.policy,
        "attn_layers": args.attn_layers,
        "selected_layers": selected_layers,
        "case_count": len(records_out),
        "run_seconds": float(time.perf_counter() - run_start),
        "cases": records_out,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "run_complete", "case_count": len(records_out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
