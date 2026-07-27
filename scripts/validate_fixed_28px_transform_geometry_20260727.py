#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from vlmeval.vlm.replay_image_transform import (
    _resize_to_profile_processed_size,
    apply_image_transform_to_content,
    canonicalize_image_transform,
    resolve_image_transform_profile,
)


TRANSFORM = "shift_right_fixed_28px"
MODEL_FAMILIES = ("qwen2_5_vl", "minicpm45")


def make_source(path: Path, width: int = 516, height: int = 307) -> None:
    y, x = np.mgrid[:height, :width]
    array = np.stack(
        [
            (x % 251).astype(np.uint8),
            (y % 241).astype(np.uint8),
            ((3 * x + 5 * y) % 253).astype(np.uint8),
        ],
        axis=-1,
    )
    Image.fromarray(array, mode="RGB").save(path)


def validate_family(source_path: Path, output_dir: Path, family: str) -> dict:
    message = [
        {"type": "image", "value": str(source_path)},
        {"type": "text", "value": "Q1 sentinel"},
        {"type": "image", "value": str(source_path)},
        {"type": "text", "value": "Q2 sentinel"},
    ]
    transformed, record = apply_image_transform_to_content(
        message,
        transform_name=TRANSFORM,
        sample_meta={"sample_index": family},
        cache_dir=output_dir / "cache" / family,
        dataset_name="fixed_28px_geometry_smoke",
        image_position=2,
        model_family=family,
    )

    profile = resolve_image_transform_profile(family)
    with Image.open(source_path) as source:
        source = source.convert("RGB")
        processed, _ = _resize_to_profile_processed_size(
            source,
            profile=profile,
            min_pixels=None,
            max_pixels=None,
        )
    expected = np.roll(np.asarray(processed), shift=28, axis=1)
    with Image.open(transformed[2]["value"]) as shifted:
        actual = np.asarray(shifted.convert("RGB"))

    checks = {
        "canonical_transform": canonicalize_image_transform(TRANSFORM, strict=True) == TRANSFORM,
        "message_shape_preserved": [item["type"] for item in transformed] == ["image", "text", "image", "text"],
        "i1_unchanged": transformed[0] == message[0],
        "q1_unchanged": transformed[1] == message[1],
        "q2_unchanged": transformed[3] == message[3],
        "i2_replaced": transformed[2]["value"] != message[2]["value"],
        "target_is_i2": record.get("target_image_position") == 2 and record.get("content_item_index") == 2,
        "family_recorded": record.get("shift", {}).get("model_family") == family,
        "fixed_processed_semantics": record.get("shift", {}).get("semantic_unit") == "fixed_processed_pixels",
        "fixed_delta": record.get("shift", {}).get("dx") == 28 and record.get("shift", {}).get("dy") == 0,
        "processed_delta_recorded": record.get("shift", {}).get("processed_shift_pixels") == 28,
        "wrap_verified": record.get("shift", {}).get("border_wrap_verified") is True,
        "exact_pixel_roll": actual.shape == expected.shape and np.array_equal(actual, expected),
    }
    return {
        "model_family": family,
        "checks": checks,
        "all_passed": all(checks.values()),
        "record": record,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "source.png"
    make_source(source_path)

    results = [validate_family(source_path, output_dir, family) for family in MODEL_FAMILIES]
    summary = {
        "contract": {
            "topology": "image_text_image_text",
            "target": "I2 only",
            "transform": TRANSFORM,
            "operation": "ordinary circular image translation on the existing processed-image path",
            "processed_shift_pixels": 28,
        },
        "results": results,
        "all_passed": all(item["all_passed"] for item in results),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary_path)
    print(json.dumps({"all_passed": summary["all_passed"]}, ensure_ascii=False))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
