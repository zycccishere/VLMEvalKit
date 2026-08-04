#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_MODELS = {
    "qwen25vl_32b": "qwen2_5_vl",
    "qwen25vl_3b": "qwen2_5_vl",
    "minicpm_o_45": "minicpm45",
}
TRANSFORM = "noise_image_seed17"
SEED = 17


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def strip_file_scheme(value: str) -> str:
    return value[len("file://") :] if value.startswith("file://") else value


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


def validate_raw_artifact(record: dict) -> dict[str, bool]:
    original_path = Path(strip_file_scheme(str(record["original_image_ref"])))
    transformed_path = Path(strip_file_scheme(str(record["transformed_image_ref"])))
    with Image.open(original_path) as original, Image.open(transformed_path) as transformed:
        original = original.convert("RGB")
        transformed = transformed.convert("RGB")
        expected = independent_noise(original.size, SEED)
        actual = np.asarray(transformed)
    noise = record.get("noise", {})
    return {
        "raw_artifact_exact": np.array_equal(actual, np.asarray(expected)),
        "raw_artifact_grayscale": np.array_equal(actual[:, :, 0], actual[:, :, 1])
        and np.array_equal(actual[:, :, 0], actual[:, :, 2]),
        "raw_artifact_nonconstant": float(actual[:, :, 0].std()) > 20.0,
        "output_hash_matches": noise.get("output_sha256") == image_sha256(transformed),
        "base_hash_matches": noise.get("base_sha256") == image_sha256(independent_noise((256, 256), SEED)),
    }


def validate_payload_pixels(record: dict, sizes: list, hashes: list) -> dict[str, bool]:
    original_path = Path(strip_file_scheme(str(record["original_image_ref"])))
    transformed_path = Path(strip_file_scheme(str(record["transformed_image_ref"])))
    if len(sizes) != 2 or len(hashes) != 2:
        return {
            "payload_has_two_pixel_hashes": False,
            "payload_i1_matches_source_pixels": False,
            "payload_i2_matches_noise_pixels": False,
            "payload_pixel_order_is_i1_i2": False,
        }
    with Image.open(original_path) as original, Image.open(transformed_path) as transformed:
        original = original.convert("RGB").resize(tuple(sizes[0]), Image.Resampling.BICUBIC)
        transformed = transformed.convert("RGB").resize(tuple(sizes[1]), Image.Resampling.BICUBIC)
    expected_i1 = image_sha256(original)
    expected_i2 = image_sha256(transformed)
    return {
        "payload_has_two_pixel_hashes": True,
        "payload_i1_matches_source_pixels": hashes[0] == expected_i1,
        "payload_i2_matches_noise_pixels": hashes[1] == expected_i2,
        "payload_pixel_order_is_i1_i2": hashes == [expected_i1, expected_i2],
    }


def validate_qwen(records: list[dict], transform: dict) -> dict[str, bool]:
    prepared = next(item for item in records if item.get("phase") == "prepared")
    processor = next(item for item in records if item.get("phase") == "processor_inputs")
    replayed = prepared.get("message_replayed", [])
    refs = prepared.get("replayed_image_refs", [])
    spans = processor.get("image_token_spans", [])
    slices = processor.get("processor_image_slices", [])
    record = transform["record"]
    checks = {
        "itit_message_order": [item.get("type") for item in replayed] == ["image", "text", "image", "text"],
        "duplicated_question": replayed[1].get("text") == replayed[3].get("text"),
        "two_distinct_image_refs": len(refs) == 2 and refs[0] != refs[1],
        "processor_has_two_images": prepared.get("vision_extract_image_count") == 2 and len(slices) == 2,
        "processor_has_two_spans": [item.get("image_position") for item in spans] == [1, 2],
        "target_span_is_i2": processor.get("target_image_span", {}).get("image_position") == 2,
        "i1_ref_preserved": refs[0].endswith(strip_file_scheme(str(record["original_image_ref"]))),
        "i2_ref_is_noise": refs[1].endswith(strip_file_scheme(str(record["transformed_image_ref"]))),
    }
    checks.update(
        validate_payload_pixels(
            record,
            prepared.get("vllm_payload_image_sizes", []),
            prepared.get("vllm_payload_image_rgb_sha256", []),
        )
    )
    return checks


def validate_minicpm(records: list[dict], transform: dict) -> dict[str, bool]:
    payload = next(item for item in records if item.get("phase") == "minicpm_vllm_payload")
    replayed = payload.get("message_replayed", [])
    record = transform["record"]
    checks = {
        "itit_message_order": [item.get("type") for item in replayed] == ["image", "text", "image", "text"],
        "duplicated_question": replayed[1].get("value") == replayed[3].get("value"),
        "vllm_payload_has_two_images": payload.get("payload_image_count") == 2,
        "vllm_payload_has_two_sizes": len(payload.get("payload_image_sizes", [])) == 2,
        "i1_ref_preserved": replayed[0].get("value") == strip_file_scheme(str(record["original_image_ref"])),
        "i2_ref_is_noise": replayed[2].get("value") == strip_file_scheme(str(record["transformed_image_ref"])),
    }
    checks.update(
        validate_payload_pixels(
            record,
            payload.get("payload_image_sizes", []),
            payload.get("payload_image_rgb_sha256", []),
        )
    )
    return checks


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
        noise = record.get("noise", {})
        checks.update(
            {
                "transform_name": transform.get("image_transform") == TRANSFORM,
                "transform_applied": record.get("applied") is True,
                "targets_i2_only": record.get("target_image_position") == 2 and record.get("content_item_index") == 2,
                "noise_seed": noise.get("seed") == SEED,
                "noise_family": noise.get("family") == "grayscale_power_spectrum_1_over_f_squared",
                "global_condition_seed": noise.get("seed_scope") == "global_condition",
            }
        )
        checks.update(validate_raw_artifact(record))
        checks.update(validate_qwen(records, transform) if family == "qwen2_5_vl" else validate_minicpm(records, transform))
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
        "contract": "ITIT; deterministic seed-17 1/f^2 noise replaces I2 only; decoder sees the real two-image payload",
        "results": results,
        "all_passed": len(results) == len(EXPECTED_MODELS) and all(item["all_passed"] for item in results),
    }
    output_path = run_root / "runtime_smoke_validation.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_path)
    print(json.dumps({"all_passed": summary["all_passed"], "models": [item["model_key"] for item in results]}))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
