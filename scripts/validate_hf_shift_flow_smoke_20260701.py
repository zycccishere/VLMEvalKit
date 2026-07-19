#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_BY_FAMILY = {
    "gemma3": {
        "image_tokens": 256,
        "stride": 56,
        "transforms": {
            "shift_right_half_vit_token": {"dx": 7, "base_pixels": 14, "semantic_unit": "vit_patch"},
            "shift_right_one_vit_token": {"dx": 14, "base_pixels": 14, "semantic_unit": "vit_patch"},
            "shift_right_one_llm_token": {"dx": 56, "base_pixels": 56, "semantic_unit": "llm_visual_token"},
        },
    },
    "minicpm-v-4_5": {
        "image_tokens": 64,
        "stride": 56,
        "transforms": {
            "shift_right_half_vit_token": {"dx": 7, "base_pixels": 14, "semantic_unit": "vit_patch"},
            "shift_right_one_vit_token": {"dx": 14, "base_pixels": 14, "semantic_unit": "vit_patch"},
            "shift_right_one_llm_token": {
                "dx": 56,
                "base_pixels": 56,
                "semantic_unit": "resampler_query_equal_area_nominal_scale",
            },
        },
    },
    "minicpm-o-4_5": {
        "image_tokens": 64,
        "stride": 56,
        "transforms": {
            "shift_right_half_vit_token": {"dx": 7, "base_pixels": 14, "semantic_unit": "vit_patch"},
            "shift_right_one_vit_token": {"dx": 14, "base_pixels": 14, "semantic_unit": "vit_patch"},
            "shift_right_one_llm_token": {
                "dx": 56,
                "base_pixels": 56,
                "semantic_unit": "resampler_query_equal_area_nominal_scale",
            },
        },
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate HF Gemma3/MiniCPM scalar shift-flow smoke artifacts.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-cases", type=int, default=0)
    parser.add_argument("--require-scalar-raw", action="store_true")
    parser.add_argument("--row-sum-atol", type=float, default=2e-3)
    parser.add_argument("--metric-atol", type=float, default=5e-3)
    return parser


def add_check(checks: list[dict[str, Any]], failures: list[dict[str, Any]], name: str, ok: bool, detail: Any) -> None:
    item = {"name": name, "ok": bool(ok), "detail": detail}
    checks.append(item)
    if not ok:
        failures.append({"name": name, "detail": detail})


def is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def metric_matches(observed: Any, expected: float, atol: float) -> bool:
    try:
        observed_float = float(observed)
    except Exception:
        return False
    return bool(np.isfinite(observed_float) and np.isfinite(expected) and abs(observed_float - expected) <= atol)


def scalar_mean_matches(observed: Any, values: np.ndarray, atol: float) -> bool:
    if values.size == 0:
        return False
    return metric_matches(observed, float(np.asarray(values, dtype=np.float64).mean()), atol)


def validate(
    output_dir: Path,
    *,
    expected_cases: int,
    require_scalar_raw: bool,
    row_sum_atol: float,
    metric_atol: float,
) -> dict[str, Any]:
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    family = str(summary.get("model_family"))
    if family not in EXPECTED_BY_FAMILY:
        raise ValueError(f"Unsupported model_family in summary: {family}")
    expected = EXPECTED_BY_FAMILY[family]
    expected_transforms = {"baseline", *expected["transforms"].keys()}
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if expected_cases:
        add_check(checks, failures, "case_count", summary.get("case_count") == expected_cases, summary.get("case_count"))

    for case in summary.get("cases", []):
        transforms = case.get("transforms", {})
        add_check(
            checks,
            failures,
            f"{case['case_id']} transform_set",
            set(transforms) == expected_transforms,
            sorted(transforms),
        )
        for transform, payload in transforms.items():
            image1_tokens = int(payload["image1_grid"]["token_count"])
            image2_tokens = int(payload["image2_grid"]["token_count"])
            span_meta = payload.get("span_meta") or {}
            token_detail = {
                "image1_tokens": image1_tokens,
                "image2_tokens": image2_tokens,
                "span_meta": span_meta,
            }
            add_check(
                checks,
                failures,
                f"{case['case_id']} {transform} image_token_count",
                image1_tokens == expected["image_tokens"] and image2_tokens == expected["image_tokens"],
                token_detail,
            )
            add_check(
                checks,
                failures,
                f"{case['case_id']} {transform} two_image_spans",
                int(span_meta.get("image_bound_count", 0)) == 2,
                span_meta,
            )
            if family == "gemma3":
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} {transform} gemma_processor_shapes",
                    span_meta.get("pixel_values_shape") == [2, 3, 896, 896]
                    and int(span_meta.get("token_type_image_count", -1)) == image1_tokens + image2_tokens
                    and span_meta.get("token_type_image_matches_spans") is True,
                    span_meta,
                )
            if family.startswith("minicpm"):
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} {transform} minicpm_position_ids",
                    span_meta.get("position_ids_shape") is not None
                    and int(span_meta.get("position_ids_first", -1)) == 0
                    and int(span_meta.get("position_ids_last", -1)) == int(span_meta.get("input_token_count", 0)) - 1,
                    span_meta,
                )
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} {transform} minicpm_processor_shapes",
                    span_meta.get("image_bound_lengths") == [expected["image_tokens"], expected["image_tokens"]]
                    and int(span_meta.get("pixel_value_count", -1)) == 2
                    and int(span_meta.get("tgt_size_count", -1)) == 2
                    and span_meta.get("tgt_sizes_values") is not None
                    and int(span_meta.get("minicpm_patch_size", -1)) == 14
                    and int(span_meta.get("minicpm_query_num", -1)) == expected["image_tokens"]
                    and (span_meta.get("minicpm_resampler_query_shape") or [0])[0] == expected["image_tokens"],
                    span_meta,
                )
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} {transform} minicpm_global_resampler_layout",
                    payload["image1_grid"].get("layout") == "global_resampler_queries"
                    and payload["image2_grid"].get("layout") == "global_resampler_queries"
                    and payload["image1_grid"].get("llm_grid_h") is None
                    and payload["image1_grid"].get("llm_grid_w") is None
                    and payload["image1_grid"].get("local_spatial_footprint") is False
                    and span_meta.get("minicpm_llm_token_has_local_footprint") is False,
                    {"image1_grid": payload["image1_grid"], "span_meta": span_meta},
                )
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} {transform} minicpm_strict_q1",
                    span_meta.get("strict_q1_positions") is True
                    and int(span_meta.get("q1_text_position_count", 0)) > 0,
                    span_meta,
                )
            record = payload.get("transform_record") or {}
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
                exp_shift = expected["transforms"][transform]
                exp_dx = int(exp_shift["dx"])
                exp_base_pixels = int(exp_shift["base_pixels"])
                shift_detail = {
                    key: shift.get(key)
                    for key in [
                        "dx",
                        "dy",
                        "processed_shift_pixels",
                        "base_pixels",
                        "semantic_unit",
                        "processed_space",
                        "border_wrap_verified",
                        "model_family",
                    ]
                }
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} {transform} shift_record",
                    shift.get("dx") == exp_dx
                    and shift.get("dy") == 0
                    and shift.get("processed_shift_pixels") == exp_dx
                    and shift.get("base_pixels") == exp_base_pixels
                    and shift.get("semantic_unit") == exp_shift["semantic_unit"]
                    and shift.get("processed_space") is True
                    and shift.get("border_wrap_verified") is True,
                    shift_detail,
                )
                meta = payload.get("content_shift_meta") or {}
                if family.startswith("minicpm"):
                    add_check(
                        checks,
                        failures,
                        f"{case['case_id']} {transform} nominal_query_pitch_units",
                        meta.get("dx_tokens") is None
                        and meta.get("llm_visual_token_stride") is None
                        and meta.get("spatial_token_mapping_valid") is False
                        and abs(
                            float(meta.get("dx_nominal_query_pitch_units", -999.0))
                            - float(exp_dx) / float(expected["stride"])
                        )
                        < 1e-8,
                        meta,
                    )
                else:
                    add_check(
                        checks,
                        failures,
                        f"{case['case_id']} {transform} dx_tokens",
                        abs(
                            float(meta.get("dx_tokens", -999.0))
                            - float(exp_dx) / float(shift.get("llm_visual_token_stride", expected["stride"]))
                        )
                        < 1e-8,
                        meta,
                    )
            for layer in payload.get("layers", []):
                layer_name = f"{case['case_id']} {transform} L{layer['layer']}"
                scalar_detail = {
                    "query_count": layer.get("query_count"),
                    "image1_key_count": layer.get("image1_key_count"),
                    "text_key_count": layer.get("text_key_count"),
                    "image2_key_count": layer.get("image2_key_count"),
                    "mass_total_max": layer.get("mass_total_max"),
                    "scalar_npz_path": layer.get("scalar_npz_path", ""),
                }
                add_check(
                    checks,
                    failures,
                    f"{layer_name} scalar_counts",
                    int(layer.get("query_count", 0)) == image2_tokens
                    and int(layer.get("image1_key_count", 0)) == image1_tokens
                    and int(layer.get("image2_key_count", 0)) == image2_tokens
                    and int(layer.get("text_key_count", 0)) > 0,
                    scalar_detail,
                )
                add_check(
                    checks,
                    failures,
                    f"{layer_name} scalar_finite",
                    is_finite_number(layer.get("mean_image1_mass_raw"))
                    and is_finite_number(layer.get("mean_text_mass_raw"))
                    and is_finite_number(layer.get("mean_image2_mass_raw"))
                    and is_finite_number(layer.get("mass_total_mean"))
                    and is_finite_number(layer.get("mass_total_max")),
                    scalar_detail,
                )
                add_check(
                    checks,
                    failures,
                    f"{layer_name} scalar_mass_leq_one",
                    float(layer.get("mass_total_max", 9.0)) <= 1.0 + row_sum_atol,
                    scalar_detail,
                )
                scalar_npz_rel = str(layer.get("scalar_npz_path", "") or "")
                add_check(
                    checks,
                    failures,
                    f"{layer_name} scalar_raw_present",
                    bool(scalar_npz_rel) or not require_scalar_raw,
                    scalar_detail,
                )
                if scalar_npz_rel:
                    scalar_path = output_dir / scalar_npz_rel
                    raw = np.load(scalar_path)
                    image1_mass = np.asarray(raw["image1_mass_raw"], dtype=np.float32)
                    text_mass = np.asarray(raw["text_mass_raw"], dtype=np.float32)
                    image2_mass = np.asarray(raw["image2_mass_raw"], dtype=np.float32)
                    raw_total = image1_mass + text_mass + image2_mass
                    raw_detail = {
                        **scalar_detail,
                        "image1_mass_shape": list(image1_mass.shape),
                        "text_mass_shape": list(text_mass.shape),
                        "image2_mass_shape": list(image2_mass.shape),
                        "raw_total_max": float(raw_total.max()) if raw_total.size else None,
                    }
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} scalar_raw_shapes",
                        image1_mass.shape == (image2_tokens,)
                        and text_mass.shape == (image2_tokens,)
                        and image2_mass.shape == (image2_tokens,),
                        raw_detail,
                    )
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} scalar_raw_finite",
                        bool(
                            np.isfinite(image1_mass).all()
                            and np.isfinite(text_mass).all()
                            and np.isfinite(image2_mass).all()
                        ),
                        raw_detail,
                    )
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} scalar_raw_metrics_match",
                        scalar_mean_matches(layer.get("mean_image1_mass_raw"), image1_mass, metric_atol)
                        and scalar_mean_matches(layer.get("mean_text_mass_raw"), text_mass, metric_atol)
                        and scalar_mean_matches(layer.get("mean_image2_mass_raw"), image2_mass, metric_atol)
                        and metric_matches(layer.get("mass_total_max"), float(raw_total.max()), metric_atol),
                        raw_detail,
                    )

    return {
        "ok": not failures,
        "check_count": len(checks),
        "failure_count": len(failures),
        "failures": failures,
        "checks_preview": checks[:50],
    }


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    result = validate(
        output_dir,
        expected_cases=args.expected_cases,
        require_scalar_raw=args.require_scalar_raw,
        row_sum_atol=args.row_sum_atol,
        metric_atol=args.metric_atol,
    )
    (output_dir / "smoke_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
