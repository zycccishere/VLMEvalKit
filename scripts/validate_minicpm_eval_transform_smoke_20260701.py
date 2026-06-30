#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vlmeval.vlm.minicpm_v_4_5_replay import MiniCPM_V_4_5_Replay
from vlmeval.vlm.replay_image_transform import canonicalize_image_transform


def make_source_image(path: Path) -> None:
    image = Image.new("RGB", (448, 448), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 80, 447], fill=(255, 0, 0))
    draw.rectangle([368, 0, 447, 447], fill=(0, 0, 255))
    draw.rectangle([180, 180, 268, 268], fill=(0, 255, 0))
    image.save(path)


def build_replay_wrapper(transform: str, cache_dir: Path, trace_file: Path) -> MiniCPM_V_4_5_Replay:
    obj = object.__new__(MiniCPM_V_4_5_Replay)
    obj.replay_cfg = {
        "mode": "image_text_image_text",
        "repeat_times": 1,
        "image_copy_mode": "reuse_path",
        "debug": False,
    }
    obj.prompt_template_cfg = {"name": "identity", "template": "{problem}"}
    obj.template_on_last_replay_text = True
    obj.image_transform_name = canonicalize_image_transform(transform)
    obj.image_transform_cache_dir = str(cache_dir)
    obj.image_transform_effective_cache_dir = str(cache_dir)
    obj.image_transform_target_position = 2
    obj._last_image_transform_record = None
    obj._minicpm_transform_records_by_message_id = {}
    obj._minicpm_trace_active_message_ids = set()
    obj._minicpm_trace_level = "summary"
    obj._minicpm_trace_max_samples = 4
    obj._minicpm_trace_seen_samples = 0
    obj._minicpm_trace_active = False
    obj._minicpm_dump_file = str(trace_file)
    obj._minicpm_dump_max_chars = 12000
    obj.debug_io_max_items = 32
    obj.debug_io_max_text_chars = 4000
    return obj


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_case(transform: str, source_path: Path, out_dir: Path) -> dict:
    case_dir = out_dir / transform
    cache_dir = case_dir / "cache"
    trace_dir = case_dir / "trace"
    cache_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_file = trace_dir / f"MiniCPM_V_4_5_Replay.pid{os.getpid()}.jsonl"

    wrapper = build_replay_wrapper(transform, cache_dir, trace_file)
    meta = {"sample_index": f"smoke_{transform}"}
    message = [
        {"type": "image", "value": str(source_path), "replay_meta": meta},
        {"type": "text", "value": "What color is the center square?", "replay_meta": meta},
    ]
    trace_active = wrapper._begin_trace_sample()
    replayed = wrapper._apply_replay_pipeline(message, dataset="SmokeDataset")
    transformed = wrapper._apply_image_transform_pipeline(
        replayed,
        inputs=message,
        dataset="SmokeDataset",
        trace_active=trace_active,
    )
    payload = wrapper._message_to_vllm_content(transformed, dataset="SmokeDataset")

    image_items = [item for item in transformed if isinstance(item, dict) and item.get("type") == "image"]
    payload_images = [item["image_pil"] for item in payload if item.get("type") == "image_pil"]
    transformed_ref = image_items[1]["value"]
    transformed_image = Image.open(wrapper._strip_file_scheme(transformed_ref)).convert("RGB")
    trace_rows = read_jsonl(trace_file)

    expected_transform = canonicalize_image_transform(transform)
    checks = {
        "replayed_iqiq_order": [item.get("type") for item in replayed] == ["image", "text", "image", "text"],
        "two_images_after_transform": len(image_items) == 2,
        "first_image_unchanged": image_items[0]["value"] == str(source_path),
        "payload_has_two_images": len(payload_images) == 2,
        "trace_has_payload": "minicpm_vllm_payload" in [row.get("phase") for row in trace_rows],
    }
    if expected_transform == "baseline":
        checks["baseline_second_image_unchanged"] = image_items[1]["value"] == str(source_path)
    else:
        checks["second_image_changed"] = image_items[1]["value"] != str(source_path)
        checks["trace_has_transform"] = "image_transform" in [row.get("phase") for row in trace_rows]
        checks["record_targets_second_image"] = wrapper._last_image_transform_record.get("target_image_position") == 2
        checks["record_item_index_is_i2"] = wrapper._last_image_transform_record.get("content_item_index") == 2
    if expected_transform == "blank":
        checks["blank_is_white"] = transformed_image.getpixel((224, 224)) == (255, 255, 255)
    if expected_transform == "shift_right_one_llm_token":
        shift = wrapper._last_image_transform_record.get("shift", {})
        checks["shift_dx_is_56"] = shift.get("dx") == 56
        checks["shift_unit_is_llm_visual_token"] = shift.get("semantic_unit") == "llm_visual_token"
        checks["wrapped_left_edge_is_old_right_edge"] = transformed_image.getpixel((0, 20)) == (0, 0, 255)

    result = {
        "transform": expected_transform,
        "checks": checks,
        "all_passed": all(checks.values()),
        "replayed_types": [item.get("type") for item in replayed],
        "transformed_image_refs": [item["value"] for item in image_items],
        "payload_image_sizes": [list(image.size) for image in payload_images],
        "trace_phases": [row.get("phase") for row in trace_rows],
        "transform_record": wrapper._last_image_transform_record,
        "trace_file": str(trace_file),
    }
    (case_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.output_dir / "source.png"
    make_source_image(source_path)
    transforms = ["baseline", "blank", "shift_right_one_llm_token"]
    results = [run_case(transform, source_path, args.output_dir) for transform in transforms]
    summary = {
        "output_dir": str(args.output_dir),
        "source_path": str(source_path),
        "case_count": len(results),
        "all_passed": all(result["all_passed"] for result in results),
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
