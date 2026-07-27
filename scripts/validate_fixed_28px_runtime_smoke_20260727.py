#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_MODELS = {
    "qwen25vl_32b": "qwen2_5_vl",
    "qwen25vl_3b": "qwen2_5_vl",
    "minicpm_o_45": "minicpm45",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def strip_file_scheme(value: str) -> str:
    return value[len("file://") :] if value.startswith("file://") else value


def validate_pixel_artifact(record: dict) -> bool:
    original_path = Path(strip_file_scheme(str(record["original_image_ref"])))
    shifted_path = Path(strip_file_scheme(str(record["transformed_image_ref"])))
    with Image.open(original_path) as original, Image.open(shifted_path) as shifted:
        original = original.convert("RGB")
        shifted = shifted.convert("RGB")
        resized = original.resize(shifted.size, resample=Image.Resampling.BICUBIC)
        expected = np.roll(np.asarray(resized), shift=28, axis=1)
        actual = np.asarray(shifted)
    return actual.shape == expected.shape and np.array_equal(actual, expected)


def validate_qwen(records: list[dict], transform: dict) -> dict[str, bool]:
    prepared = next(item for item in records if item.get("phase") == "prepared")
    processor = next(item for item in records if item.get("phase") == "processor_inputs")
    replayed = prepared.get("message_replayed", [])
    refs = prepared.get("replayed_image_refs", [])
    spans = processor.get("image_token_spans", [])
    slices = processor.get("processor_image_slices", [])
    return {
        "iqiq_message_order": [item.get("type") for item in replayed] == ["image", "text", "image", "text"],
        "duplicated_question": replayed[1].get("text") == replayed[3].get("text"),
        "two_distinct_image_refs": len(refs) == 2 and refs[0] != refs[1],
        "processor_has_two_images": prepared.get("vision_extract_image_count") == 2 and len(slices) == 2,
        "processor_has_two_spans": [item.get("image_position") for item in spans] == [1, 2],
        "target_span_is_i2": processor.get("target_image_span", {}).get("image_position") == 2,
        "i1_ref_preserved": refs[0].endswith(strip_file_scheme(str(transform["record"]["original_image_ref"]))),
        "i2_ref_is_transformed": refs[1].endswith(strip_file_scheme(str(transform["record"]["transformed_image_ref"]))),
    }


def validate_minicpm(records: list[dict], transform: dict) -> dict[str, bool]:
    payload = next(item for item in records if item.get("phase") == "minicpm_vllm_payload")
    replayed = payload.get("message_replayed", [])
    original_ref = strip_file_scheme(str(transform["record"]["original_image_ref"]))
    transformed_ref = strip_file_scheme(str(transform["record"]["transformed_image_ref"]))
    return {
        "iqiq_message_order": [item.get("type") for item in replayed] == ["image", "text", "image", "text"],
        "duplicated_question": replayed[1].get("value") == replayed[3].get("value"),
        "vllm_payload_has_two_images": payload.get("payload_image_count") == 2,
        "vllm_payload_has_two_sizes": len(payload.get("payload_image_sizes", [])) == 2,
        "i1_ref_preserved": replayed[0].get("value") == original_ref,
        "i2_ref_is_transformed": replayed[2].get("value") == transformed_ref,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()

    results = []
    for model_key, family in EXPECTED_MODELS.items():
        trace_files = list(run_root.glob(f"**/{model_key}/**/_trace/*.jsonl"))
        checks: dict[str, bool] = {"single_trace_file": len(trace_files) == 1}
        if len(trace_files) != 1:
            results.append({"model_key": model_key, "checks": checks, "all_passed": False})
            continue
        records = load_jsonl(trace_files[0])
        transforms = [item for item in records if item.get("phase") == "image_transform"]
        checks["single_transform_record"] = len(transforms) == 1
        if len(transforms) != 1:
            results.append({"model_key": model_key, "trace_file": str(trace_files[0]), "checks": checks, "all_passed": False})
            continue
        transform = transforms[0]
        record = transform.get("record", {})
        shift = record.get("shift", {})
        checks.update(
            {
                "transform_name": transform.get("image_transform") == "shift_right_fixed_28px",
                "transform_applied": record.get("applied") is True,
                "targets_i2_only": record.get("target_image_position") == 2 and record.get("content_item_index") == 2,
                "fixed_processed_semantics": shift.get("semantic_unit") == "fixed_processed_pixels",
                "family_recorded": shift.get("model_family") == family,
                "fixed_delta": shift.get("dx") == 28 and shift.get("dy") == 0,
                "wrap_verified": shift.get("border_wrap_verified") is True,
                "raw_artifact_exact": validate_pixel_artifact(record),
            }
        )
        if family == "qwen2_5_vl":
            checks.update(validate_qwen(records, transform))
        else:
            checks.update(validate_minicpm(records, transform))
        results.append(
            {
                "model_key": model_key,
                "model_family": family,
                "trace_file": str(trace_files[0]),
                "checks": checks,
                "all_passed": all(checks.values()),
            }
        )

    summary = {
        "run_root": str(run_root),
        "contract": "IQIQ; I2-only; ordinary processed-image circular shift right by 28 pixels",
        "results": results,
        "all_passed": len(results) == len(EXPECTED_MODELS) and all(item["all_passed"] for item in results),
    }
    output_path = run_root / "runtime_smoke_validation.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_path)
    print(json.dumps({"all_passed": summary["all_passed"], "models": [r["model_key"] for r in results]}))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
