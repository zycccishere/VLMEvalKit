#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "vlmeval" / "vlm" / "replay_image_transform.py"


def load_transform_module() -> Any:
    spec = importlib.util.spec_from_file_location("replay_image_transform_direct", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load transform module: {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_source_image(path: Path) -> None:
    image = Image.new("RGB", (516, 307), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 515, 80], fill=(255, 255, 0))
    draw.rectangle([0, 226, 515, 306], fill=(255, 0, 255))
    draw.rectangle([0, 0, 80, 306], fill=(255, 0, 0))
    draw.rectangle([435, 0, 515, 306], fill=(0, 0, 255))
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    module = load_transform_module()
    source_path = args.output_dir / "source.png"
    make_source_image(source_path)
    content = [
        {"type": "image", "value": str(source_path)},
        {"type": "text", "value": "q"},
        {"type": "image", "value": str(source_path)},
        {"type": "text", "value": "q"},
    ]
    cases = {
        "shift_right_half_vit_token": {"delta": (7, 0)},
        "shift_right_one_vit_token": {"delta": (14, 0)},
        "shift_right_one_llm_token": {"delta": (56, 0), "probe": ((0, 150), (0, 0, 255))},
        "shift_left_one_llm_token": {"delta": (-56, 0), "probe": ((573, 150), (255, 0, 0))},
        "shift_down_one_llm_token": {"delta": (0, 56), "probe": ((280, 0), (255, 0, 255))},
        "shift_up_one_llm_token": {"delta": (0, -56), "probe": ((280, 349), (255, 255, 0))},
    }
    results = []
    for transform, expected in cases.items():
        transformed, record = module.apply_image_transform_to_content(
            content,
            transform_name=transform,
            sample_meta={"sample_index": transform},
            cache_dir=args.output_dir / "cache",
            dataset_name="geometry_smoke",
            image_position=2,
            model_family="minicpm45",
        )
        shift = record["shift"]
        checks = {
            "targets_i2": record["target_image_position"] == 2 and record["content_item_index"] == 2,
            "delta": (shift["dx"], shift["dy"]) == expected["delta"],
            "processed_size": record["processed_image_size_before_shift"] == [574, 350],
            "vit_grid": (shift["processed_vit_grid_h"], shift["processed_vit_grid_w"]) == (25, 41),
            "no_fake_llm_grid": shift["processed_llm_grid_h"] is None and shift["processed_llm_grid_w"] is None,
            "global_resampler": shift["minicpm45_llm_spatial_layout"] == "global_resampler_queries",
            "no_local_footprint": shift["minicpm45_llm_token_has_local_footprint"] is False,
            "nominal_pitch": shift["minicpm45_nominal_query_pitch"] == 56,
            "border_wrap": shift["border_wrap_verified"] is True,
        }
        if "llm" in transform:
            checks["nominal_semantic_unit"] = (
                shift["semantic_unit"] == "resampler_query_equal_area_nominal_scale"
                and shift["llm_visual_token_stride"] is None
            )
        else:
            checks["vit_semantic_unit"] = shift["semantic_unit"] == "vit_patch" and shift["vit_patch_size"] == 14
        if "probe" in expected:
            image_ref = module._strip_file_scheme(transformed[2]["value"])
            with Image.open(image_ref) as image:
                point, color = expected["probe"]
                checks["wrapped_edge_pixels"] = image.convert("RGB").getpixel(point) == color
        results.append({"transform": transform, "checks": checks, "record": record, "all_passed": all(checks.values())})

    summary = {"all_passed": all(item["all_passed"] for item in results), "results": results}
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
