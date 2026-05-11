#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from qwen25vl_image2_probe import (  # noqa: E402
    RopeAlignPatch,
    build_inputs,
    find_image_spans,
    load_model_and_processor,
    resolve_input_device,
    sanitize_single_process_env,
    set_seed,
    tensor_to_device,
)
from vlmeval.cross_image_flow import (  # noqa: E402
    CrossImageAttentionTracer,
    box_to_patch_indices,
    build_image_grid_metas,
    build_patch_boxes,
    clamp_box_to_image,
    downsample_cross_image_map,
    exact_diagonal_indices,
    mean_records_by_layer,
    round_tensor,
    summarize_cross_image_map,
    get_visual_backbone,
)
from vlmeval.probe_attention import parse_attention_layers  # noqa: E402
from vlmeval.vlm.replay_policy import apply_replay, canonicalize_replay_mode  # noqa: E402


@dataclass
class ControlItem:
    id: str
    group_id: str
    group_type: str
    source_dataset: str
    source_row_index: int
    image_path: Path
    question: str
    target_label: str
    target_box: dict[str, float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Qwen2.5-VL IIT vs ITI patch-level cross-image flow maps from a controlled image-question manifest."
    )
    parser.add_argument(
        "--model-path",
        default="/models/Qwen2.5-VL-32B-Instruct",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--item-ids", nargs="*")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["image_image_text", "image_text_image"],
        help="Replay modes to compare. Default exports IIT and ITI.",
    )
    parser.add_argument(
        "--attn-layers",
        default="last4",
        help="Attention layers to average for the exported query-to-key maps.",
    )
    parser.add_argument(
        "--viewer-max-side",
        type=int,
        default=18,
        help="Maximum patch-grid side length serialized into the viewer-facing JSON.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--rope-align", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)
    return parser


def load_control_items(manifest_path: Path, *, item_ids: set[str] | None, max_items: int) -> tuple[dict[str, Any], list[ControlItem]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items: list[ControlItem] = []
    for raw in payload["items"]:
        item = ControlItem(
            id=str(raw["id"]),
            group_id=str(raw["group_id"]),
            group_type=str(raw["group_type"]),
            source_dataset=str(raw["source_dataset"]),
            source_row_index=int(raw["source_row_index"]),
            image_path=(manifest_path.parent / raw["image_file"]).resolve(),
            question=str(raw["question"]),
            target_label=str(raw["target_label"]),
            target_box={k: float(v) for k, v in raw["target_box"].items()},
        )
        if item_ids and item.id not in item_ids:
            continue
        items.append(item)
    if max_items > 0:
        items = items[:max_items]
    return payload, items


def build_controlled_replayed_content(item: ControlItem, mode: str) -> list[dict[str, Any]]:
    base_content = [
        {"type": "image", "image": str(item.image_path)},
        {"type": "text", "text": item.question},
    ]
    return apply_replay(
        base_content,
        mode=canonicalize_replay_mode(mode),
        repeat_times=1,
        image_copy_mode="reuse_path",
    )


def maybe_generate_text(
    *,
    model,
    processor,
    model_inputs: dict[str, Any],
    prompt_len: int,
    max_new_tokens: int,
) -> dict[str, Any] | None:
    if max_new_tokens <= 0:
        return None
    with torch.inference_mode():
        sequences = model.generate(
            **model_inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=max_new_tokens,
        )
    generated_ids = sequences[0, prompt_len:].detach().cpu().tolist()
    generated_text = processor.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    del sequences
    torch.cuda.empty_cache()
    return {
        "generated_token_count": len(generated_ids),
        "generated_text": generated_text,
    }


def build_mode_record(
    *,
    item: ControlItem,
    mode: str,
    processor,
    model,
    tracer: CrossImageAttentionTracer,
    selected_layers: list[int],
    device: str,
    max_new_tokens: int,
    viewer_max_side: int,
) -> dict[str, Any]:
    content = build_controlled_replayed_content(item, mode)
    _, prompt_text, model_inputs = build_inputs(processor, content)
    input_ids = model_inputs["input_ids"][0].tolist()
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_spans = find_image_spans(input_ids, image_token_id)
    if len(image_spans) != 2:
        raise ValueError(f"Expected exactly two image spans for mode={mode}, got {len(image_spans)}.")

    visual = get_visual_backbone(model.model if hasattr(model, "model") else model)
    image_grid_thw = model_inputs.get("image_grid_thw")
    image_grid_metas = build_image_grid_metas(image_grid_thw, int(visual.spatial_merge_size))
    if len(image_grid_metas) < 2:
        raise ValueError(f"Expected at least two image grids, got {len(image_grid_metas)}.")

    query_positions = list(range(image_spans[1].start, image_spans[1].end + 1))
    key_positions = list(range(image_spans[0].start, image_spans[0].end + 1))
    tracer.set_trace_positions(query_positions=query_positions, key_positions=key_positions)
    tracer.reset()

    image_width, image_height = Image.open(item.image_path).size
    target_box = clamp_box_to_image(item.target_box, image_width, image_height)
    image1_patches = build_patch_boxes(
        width=image_width,
        height=image_height,
        grid_h=image_grid_metas[0].llm_grid_h,
        grid_w=image_grid_metas[0].llm_grid_w,
    )
    image2_patches = build_patch_boxes(
        width=image_width,
        height=image_height,
        grid_h=image_grid_metas[1].llm_grid_h,
        grid_w=image_grid_metas[1].llm_grid_w,
    )
    target_key_indices = box_to_patch_indices(box=target_box, patch_boxes=image1_patches)
    target_query_indices = box_to_patch_indices(box=target_box, patch_boxes=image2_patches)
    if not target_key_indices:
        raise ValueError(f"Target box for {item.id} did not overlap any image1 patches.")

    model_inputs = tensor_to_device(model_inputs, device)
    with torch.inference_mode():
        _ = model(**model_inputs, use_cache=False, return_dict=True)

    _per_layer_maps, mean_map = mean_records_by_layer(tracer.records, selected_layers)
    image1_cross_mass = mean_map.sum(dim=-1)
    normalized_map = mean_map / image1_cross_mass.unsqueeze(-1).clamp_min(1e-8)
    diag_indices = exact_diagonal_indices(image2_patches=image2_patches, image1_patches=image1_patches)
    summary_metrics = summarize_cross_image_map(
        q_to_k=mean_map,
        target_query_indices=target_query_indices,
        target_key_indices=target_key_indices,
        exact_diag_indices=diag_indices,
    )

    viewer_mean_map, viewer_query_grid, viewer_key_grid = downsample_cross_image_map(
        q_to_k=mean_map,
        query_grid_h=image_grid_metas[1].llm_grid_h,
        query_grid_w=image_grid_metas[1].llm_grid_w,
        key_grid_h=image_grid_metas[0].llm_grid_h,
        key_grid_w=image_grid_metas[0].llm_grid_w,
        max_side=viewer_max_side,
    )
    viewer_cross_mass = viewer_mean_map.sum(dim=-1)
    viewer_normalized = viewer_mean_map / viewer_cross_mass.unsqueeze(-1).clamp_min(1e-8)
    viewer_image1_patches = build_patch_boxes(
        width=image_width,
        height=image_height,
        grid_h=viewer_key_grid[0],
        grid_w=viewer_key_grid[1],
    )
    viewer_image2_patches = build_patch_boxes(
        width=image_width,
        height=image_height,
        grid_h=viewer_query_grid[0],
        grid_w=viewer_query_grid[1],
    )
    viewer_target_key_indices = box_to_patch_indices(box=target_box, patch_boxes=viewer_image1_patches)
    viewer_target_query_indices = box_to_patch_indices(box=target_box, patch_boxes=viewer_image2_patches)

    generated = maybe_generate_text(
        model=model,
        processor=processor,
        model_inputs=model_inputs,
        prompt_len=len(input_ids),
        max_new_tokens=max_new_tokens,
    )

    return {
        "mode": mode,
        "prompt_text": prompt_text,
        "input_token_count": len(input_ids),
        "image_spans": [{"start": span.start, "end": span.end, "length": span.length} for span in image_spans],
        "image_grid_metas": [meta.__dict__ for meta in image_grid_metas[:2]],
        "image_size": {"width": image_width, "height": image_height},
        "target_box": target_box,
        "raw_grid_shape": {
            "image1": {"h": image_grid_metas[0].llm_grid_h, "w": image_grid_metas[0].llm_grid_w},
            "image2": {"h": image_grid_metas[1].llm_grid_h, "w": image_grid_metas[1].llm_grid_w},
        },
        "viewer_grid_shape": {
            "image1": {"h": viewer_key_grid[0], "w": viewer_key_grid[1]},
            "image2": {"h": viewer_query_grid[0], "w": viewer_query_grid[1]},
        },
        "target_query_patch_indices": viewer_target_query_indices,
        "target_key_patch_indices": viewer_target_key_indices,
        "image1_patch_boxes": viewer_image1_patches,
        "image2_patch_boxes": viewer_image2_patches,
        "selected_layers": selected_layers,
        "normalized_q_to_k": round_tensor(viewer_normalized, digits=6),
        "summary_metrics": summary_metrics,
        "generated": generated,
    }


def main() -> int:
    args = build_parser().parse_args()
    sanitize_single_process_env()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest).resolve()
    manifest_payload, items = load_control_items(
        manifest_path,
        item_ids=set(args.item_ids) if args.item_ids else None,
        max_items=int(args.max_items),
    )
    if not items:
        raise ValueError("No controlled-set items matched the requested filters.")

    processor, model = load_model_and_processor(args.model_path, args.device)
    input_device = resolve_input_device(model, args.device)
    selected_layers = parse_attention_layers(args.attn_layers, len(model.model.language_model.layers))
    tracer = CrossImageAttentionTracer(
        {layer_idx: model.model.language_model.layers[layer_idx].self_attn for layer_idx in selected_layers}
    )
    tracer.patch()

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    rope_patch = RopeAlignPatch(model, image_token_id, enabled=args.rope_align)
    rope_patch.__enter__()

    summary_items: list[dict[str, Any]] = []
    try:
        for item in items:
            case_payload = {
                "id": item.id,
                "group_id": item.group_id,
                "group_type": item.group_type,
                "source_dataset": item.source_dataset,
                "source_row_index": item.source_row_index,
                "image_path": str(item.image_path),
                "question": item.question,
                "target_label": item.target_label,
                "target_box": item.target_box,
                "modes": {},
            }
            for mode in args.modes:
                tracer.reset()
                mode_record = build_mode_record(
                    item=item,
                    mode=mode,
                    processor=processor,
                    model=model,
                    tracer=tracer,
                    selected_layers=selected_layers,
                    device=input_device,
                    max_new_tokens=args.max_new_tokens,
                    viewer_max_side=args.viewer_max_side,
                )
                case_payload["modes"][mode] = mode_record

            (cases_dir / f"{item.id}.json").write_text(
                json.dumps(case_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            summary_items.append(
                {
                    "id": item.id,
                    "group_id": item.group_id,
                    "group_type": item.group_type,
                    "question": item.question,
                    "target_label": item.target_label,
                    "modes": {
                        mode: case_payload["modes"][mode]["summary_metrics"] for mode in case_payload["modes"]
                    },
                }
            )
            print(
                json.dumps(
                    {
                        "event": "case_complete",
                        "id": item.id,
                        "modes": list(case_payload["modes"]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        tracer.restore()
        rope_patch.__exit__(None, None, None)

    summary = {
        "manifest_id": manifest_payload["id"],
        "manifest_path": str(manifest_path),
        "item_count": len(summary_items),
        "modes": args.modes,
        "selected_layers": selected_layers,
        "rope_align": bool(args.rope_align),
        "items": summary_items,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
