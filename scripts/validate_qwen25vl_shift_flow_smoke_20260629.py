#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_SHIFT = {
    "shift_right_half_vit_token": {
        "dx": 7,
        "processed_shift_pixels": 7,
        "base_pixels": 14,
        "semantic_unit": "vit_patch",
        "dx_tokens": 0.25,
    },
    "shift_right_one_vit_token": {
        "dx": 14,
        "processed_shift_pixels": 14,
        "base_pixels": 14,
        "semantic_unit": "vit_patch",
        "dx_tokens": 0.5,
    },
    "shift_right_one_llm_token": {
        "dx": 28,
        "processed_shift_pixels": 28,
        "base_pixels": 28,
        "semantic_unit": "llm_visual_token",
        "dx_tokens": 1.0,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate raw smoke artifacts from qwen25vl_shift_flow_probe.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-cases", type=int, default=0)
    parser.add_argument("--row-sum-atol", type=float, default=2e-3)
    return parser


def add_check(checks: list[dict[str, Any]], failures: list[dict[str, Any]], name: str, ok: bool, detail: Any) -> None:
    item = {"name": name, "ok": bool(ok), "detail": detail}
    checks.append(item)
    if not ok:
        failures.append({"name": name, "detail": detail})


def validate(output_dir: Path, *, expected_cases: int, row_sum_atol: float) -> dict[str, Any]:
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if expected_cases:
        add_check(checks, failures, "case_count", summary.get("case_count") == expected_cases, summary.get("case_count"))

    for case in summary["cases"]:
        transforms = case["transforms"]
        add_check(
            checks,
            failures,
            f"{case['case_id']} transform_set",
            set(transforms) == {"baseline", *EXPECTED_SHIFT.keys()},
            sorted(transforms),
        )
        baseline_tokens = None
        if "baseline" in transforms:
            baseline_tokens = (
                int(transforms["baseline"]["image1_grid"]["token_count"]),
                int(transforms["baseline"]["image2_grid"]["token_count"]),
            )
        for transform, payload in transforms.items():
            image1_tokens = int(payload["image1_grid"]["token_count"])
            image2_tokens = int(payload["image2_grid"]["token_count"])
            add_check(
                checks,
                failures,
                f"{case['case_id']} {transform} repeated_image_token_count_match",
                image1_tokens == image2_tokens,
                {"image1_tokens": image1_tokens, "image2_tokens": image2_tokens},
            )
            if baseline_tokens is not None:
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} {transform} token_count_matches_baseline",
                    (image1_tokens, image2_tokens) == baseline_tokens,
                    {"baseline": baseline_tokens, "current": (image1_tokens, image2_tokens)},
                )
            record = payload["transform_record"]
            if transform == "baseline":
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} baseline_not_applied",
                    record.get("applied") is False,
                    record,
                )
            else:
                shift = record.get("shift") or {}
                expected = EXPECTED_SHIFT[transform]
                detail = {
                    key: shift.get(key)
                    for key in [
                        "dx",
                        "dy",
                        "processed_shift_pixels",
                        "base_pixels",
                        "semantic_unit",
                        "processed_space",
                        "border_wrap_verified",
                    ]
                }
                ok = (
                    shift.get("dx") == expected["dx"]
                    and shift.get("dy") == 0
                    and shift.get("processed_shift_pixels") == expected["processed_shift_pixels"]
                    and shift.get("base_pixels") == expected["base_pixels"]
                    and shift.get("semantic_unit") == expected["semantic_unit"]
                    and shift.get("processed_space") is True
                    and shift.get("border_wrap_verified") is True
                )
                add_check(checks, failures, f"{case['case_id']} {transform} shift_record", ok, detail)
                meta = payload.get("content_shift_meta") or {}
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} {transform} dx_tokens",
                    abs(float(meta.get("dx_tokens", -999.0)) - expected["dx_tokens"]) < 1e-8,
                    meta,
                )
            for layer in payload["layers"]:
                layer_name = f"{case['case_id']} {transform} L{layer['layer']}"
                npz_path = output_dir / layer["npz_path"]
                data = np.load(npz_path)
                matrix_norm = np.asarray(data["matrix_norm"], dtype=np.float32)
                matrix_raw = np.asarray(data["matrix_raw"], dtype=np.float32)
                image2_block = np.asarray(data["image2_block_raw"], dtype=np.float32)
                image1_mass = np.asarray(data["image1_mass_raw"], dtype=np.float32)
                text_mass = np.asarray(data["text_mass_raw"], dtype=np.float32)
                image2_mass = np.asarray(data["image2_mass_raw"], dtype=np.float32)
                token_ids = np.arange(image2_block.shape[0])
                future_mask = token_ids[None, :] > token_ids[:, None]
                row_error = float(np.max(np.abs(matrix_norm.sum(axis=-1) - 1.0)))
                raw_mass_error = float(np.max(np.abs(matrix_raw.sum(axis=-1) - image1_mass)))
                image2_mass_error = float(np.max(np.abs(image2_block.sum(axis=-1) - image2_mass)))
                future_mass_max = (
                    float(np.max(np.where(future_mask, image2_block, 0.0).sum(axis=-1)))
                    if image2_block.shape[0]
                    else 0.0
                )
                mass_total_max = float(np.max(image1_mass + text_mass + image2_mass))
                add_check(
                    checks,
                    failures,
                    f"{layer_name} matrix_shape",
                    matrix_norm.shape == (image2_tokens, image1_tokens),
                    matrix_norm.shape,
                )
                add_check(
                    checks,
                    failures,
                    f"{layer_name} image2_block_shape",
                    image2_block.shape == (image2_tokens, image2_tokens),
                    image2_block.shape,
                )
                add_check(checks, failures, f"{layer_name} row_norm", row_error < row_sum_atol, row_error)
                add_check(checks, failures, f"{layer_name} raw_mass", raw_mass_error < row_sum_atol, raw_mass_error)
                add_check(checks, failures, f"{layer_name} image2_mass", image2_mass_error < row_sum_atol, image2_mass_error)
                add_check(checks, failures, f"{layer_name} future_i2_mass", future_mass_max < row_sum_atol, future_mass_max)
                add_check(checks, failures, f"{layer_name} mass_leq_one", mass_total_max <= 1.0 + row_sum_atol, mass_total_max)
                finite = (
                    np.isfinite(matrix_norm).all()
                    and np.isfinite(matrix_raw).all()
                    and np.isfinite(image2_block).all()
                    and np.isfinite(image1_mass).all()
                )
                add_check(checks, failures, f"{layer_name} finite", bool(finite), str(npz_path))

    return {
        "ok": not failures,
        "check_count": len(checks),
        "failure_count": len(failures),
        "failures": failures,
        "checks_preview": checks[:40],
    }


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    result = validate(output_dir, expected_cases=args.expected_cases, row_sum_atol=args.row_sum_atol)
    (output_dir / "smoke_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
