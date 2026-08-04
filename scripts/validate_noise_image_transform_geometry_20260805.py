#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from vlmeval.vlm.replay_image_transform import apply_image_transform_to_content


NOISE_SEEDS = {
    "noise_image_seed17": 17,
    "noise_image_seed29": 29,
    "noise_image_seed43": 43,
}
MODEL_FAMILIES = ("qwen2_5_vl", "minicpm45")


def image_sha256(image: Image.Image) -> str:
    return hashlib.sha256(np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes()).hexdigest()


def independent_noise(size: tuple[int, int], seed: int) -> Image.Image:
    width, height = size
    side = 256
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((side, side))
    fy = np.fft.fftfreq(side)[:, None]
    fx = np.fft.fftfreq(side)[None, :]
    frequency = np.sqrt(fx * fx + fy * fy)
    frequency[0, 0] = 1.0
    filtered = np.fft.ifft2(np.fft.fft2(white) / frequency).real
    filtered = (filtered - filtered.mean()) / filtered.std()
    grayscale = np.clip(127.5 + 40.0 * filtered, 24.0, 231.0).astype(np.uint8)
    rgb = np.repeat(grayscale[:, :, None], 3, axis=2)
    return Image.fromarray(rgb, mode="RGB").resize((width, height), Image.Resampling.BICUBIC)


def make_source(path: Path, width: int, height: int, offset: int) -> None:
    y, x = np.mgrid[:height, :width]
    array = np.stack(
        [
            ((x + offset) % 251).astype(np.uint8),
            ((y + 2 * offset) % 241).astype(np.uint8),
            ((3 * x + 5 * y + offset) % 253).astype(np.uint8),
        ],
        axis=-1,
    )
    Image.fromarray(array, mode="RGB").save(path)


def validate_case(
    i1_path: Path,
    i2_path: Path,
    output_dir: Path,
    family: str,
    transform: str,
    seed: int,
) -> dict:
    message = [
        {"type": "image", "value": str(i1_path)},
        {"type": "text", "value": "Q1 sentinel"},
        {"type": "image", "value": str(i2_path)},
        {"type": "text", "value": "Q2 sentinel"},
    ]
    transformed, record = apply_image_transform_to_content(
        message,
        transform_name=transform,
        sample_meta={"sample_index": f"{family}_{seed}"},
        cache_dir=output_dir / "cache" / family / transform,
        dataset_name="noise_geometry_smoke",
        image_position=2,
        model_family=family,
    )

    with Image.open(i2_path) as source_i2:
        expected = independent_noise(source_i2.size, seed)
    with Image.open(transformed[2]["value"]) as observed_image:
        observed = observed_image.convert("RGB")
        observed_array = np.asarray(observed)
    expected_array = np.asarray(expected)
    noise = record.get("noise", {})
    checks = {
        "message_order_preserved": [item["type"] for item in transformed] == ["image", "text", "image", "text"],
        "i1_unchanged": transformed[0] == message[0],
        "q1_unchanged": transformed[1] == message[1],
        "q2_unchanged": transformed[3] == message[3],
        "i2_replaced": transformed[2]["value"] != message[2]["value"],
        "target_is_i2": record.get("target_image_position") == 2 and record.get("content_item_index") == 2,
        "exact_independent_reconstruction": np.array_equal(observed_array, expected_array),
        "output_size_matches_i2": list(observed.size) == record.get("original_image_size"),
        "grayscale_channels_equal": np.array_equal(observed_array[:, :, 0], observed_array[:, :, 1])
        and np.array_equal(observed_array[:, :, 0], observed_array[:, :, 2]),
        "nonconstant_noise": float(observed_array[:, :, 0].std()) > 20.0,
        "seed_recorded": noise.get("seed") == seed,
        "global_seed_scope": noise.get("seed_scope") == "global_condition",
        "family_recorded": noise.get("family") == "grayscale_power_spectrum_1_over_f_squared",
        "bicubic_recorded": noise.get("resize_interpolation") == "bicubic",
        "output_hash_matches": noise.get("output_sha256") == image_sha256(observed),
        "base_hash_matches": noise.get("base_sha256") == image_sha256(independent_noise((256, 256), seed)),
    }
    return {
        "model_family": family,
        "transform": transform,
        "seed": seed,
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
    i1_path = output_dir / "source_i1.png"
    i2_path = output_dir / "source_i2.png"
    make_source(i1_path, 83, 47, 11)
    make_source(i2_path, 73, 41, 29)

    results = [
        validate_case(i1_path, i2_path, output_dir, family, transform, seed)
        for family in MODEL_FAMILIES
        for transform, seed in NOISE_SEEDS.items()
    ]
    output_hashes = {
        item["seed"]: item["record"]["noise"]["output_sha256"]
        for item in results
        if item["model_family"] == MODEL_FAMILIES[0]
    }
    summary = {
        "contract": {
            "topology": "image_text_image_text",
            "target": "I2 only",
            "base": "256x256 deterministic grayscale 1/f^2 noise",
            "seeds": list(NOISE_SEEDS.values()),
            "resize": "bicubic to source I2 size",
        },
        "distinct_seed_outputs": len(set(output_hashes.values())) == len(NOISE_SEEDS),
        "results": results,
    }
    summary["all_passed"] = summary["distinct_seed_outputs"] and all(item["all_passed"] for item in results)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary_path)
    print(json.dumps({"all_passed": summary["all_passed"], "cases": len(results)}))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
