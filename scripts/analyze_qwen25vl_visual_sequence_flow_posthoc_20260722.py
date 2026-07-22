#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_qwen25vl_visual_sequence_flow_stats_20260722 import (
    CANONICAL_TRANSFORMS,
    paired_summary,
)


ROLL = "visual_sequence_roll_right_1"
CORE_COLUMNS = [
    "delta_mean_image1_mass_raw",
    "delta_i2_past_self_mass_raw",
    "delta_mean_text_mass_raw",
    "delta_mean_image2_mass_raw",
    "delta_row_entropy",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exploratory diagnostics for the validated Qwen visual-sequence flow run."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-cases", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--sign-flip-resamples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def require_finite(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        if not bool(np.isfinite(values).all()):
            raise ValueError(f"Non-finite values in {column}")


def validate_input(df: pd.DataFrame, expected_cases: int) -> None:
    required = {
        "case_id",
        "transform",
        "layer",
        "source_index_exact_mass",
        "exact_position_mass",
        "exact_position_mass__baseline",
        "source_minus_position_exact_mass",
        "mean_image1_mass_raw",
        "mean_image1_mass_raw__baseline",
        "i2_past_self_mass_raw",
        "i2_past_self_mass_raw__baseline",
        "delta_mass_total_mean",
        *CORE_COLUMNS,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if expected_cases <= 0:
        raise ValueError("--expected-cases must be positive")
    if sorted(df["transform"].unique().tolist()) != sorted(CANONICAL_TRANSFORMS):
        raise ValueError("Unexpected transform set")
    if sorted(df["layer"].unique().tolist()) != [60, 61, 62, 63]:
        raise ValueError("Expected Qwen32 layers [60, 61, 62, 63]")
    if int(df["case_id"].nunique()) != expected_cases:
        raise ValueError(f"Expected {expected_cases} unique cases")
    if df.duplicated(["case_id", "transform", "layer"]).any():
        raise ValueError("Duplicate case/transform/layer rows")
    roll = df[df["transform"] == ROLL]
    expected_cells = pd.Index([60, 61, 62, 63], name="layer")
    counts = roll.groupby("layer")["case_id"].nunique().reindex(expected_cells, fill_value=0)
    if not bool((counts == expected_cases).all()):
        raise ValueError(f"Incomplete roll layer cells: {counts.to_dict()}")
    require_finite(
        roll,
        [
            "source_index_exact_mass",
            "exact_position_mass",
            "exact_position_mass__baseline",
            "source_minus_position_exact_mass",
            "mean_image1_mass_raw",
            "mean_image1_mass_raw__baseline",
            "i2_past_self_mass_raw",
            "i2_past_self_mass_raw__baseline",
            "delta_mass_total_mean",
            *CORE_COLUMNS,
        ],
    )


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["source_exact_minus_baseline_slot_exact"] = (
        out["source_index_exact_mass"] - out["exact_position_mass__baseline"]
    )
    baseline_visual_denominator = (
        out["mean_image1_mass_raw__baseline"] + out["i2_past_self_mass_raw__baseline"]
    )
    roll_visual_denominator = out["mean_image1_mass_raw"] + out["i2_past_self_mass_raw"]
    if bool((baseline_visual_denominator <= 0).any() or (roll_visual_denominator <= 0).any()):
        raise ValueError("Visual-history share has a non-positive denominator")
    out["baseline_i1_visual_history_share"] = (
        out["mean_image1_mass_raw__baseline"] / baseline_visual_denominator
    )
    out["roll_i1_visual_history_share"] = out["mean_image1_mass_raw"] / roll_visual_denominator
    out["delta_i1_visual_history_share"] = (
        out["roll_i1_visual_history_share"] - out["baseline_i1_visual_history_share"]
    )
    out["delta_visual_history_mass"] = (
        out["delta_mean_image1_mass_raw"] + out["delta_i2_past_self_mass_raw"]
    )
    out["delta_i1_minus_past_self"] = (
        out["delta_mean_image1_mass_raw"] - out["delta_i2_past_self_mass_raw"]
    )
    out["component_mass_delta_sum"] = (
        out["delta_mean_image1_mass_raw"]
        + out["delta_mean_text_mass_raw"]
        + out["delta_mean_image2_mass_raw"]
    )
    out["mass_accounting_residual"] = out["component_mass_delta_sum"] - out["delta_mass_total_mean"]
    return out


def main() -> int:
    args = build_parser().parse_args()
    if args.bootstrap_resamples <= 0 or args.sign_flip_resamples <= 0:
        raise ValueError("Resample counts must be positive")
    input_path = Path(args.input_csv)
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    df = pd.read_csv(input_path)
    validate_input(df, args.expected_cases)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roll = add_derived_columns(df[df["transform"] == ROLL])
    max_layer = int(roll["layer"].max())
    last = roll[roll["layer"] == max_layer].copy()

    derived_specs = [
        ("source_exact_minus_baseline_slot_exact", "exploratory_difference_not_equivalence"),
        ("delta_i1_visual_history_share", "exploratory_relative_visual_share"),
        ("delta_visual_history_mass", "exploratory_visual_mass"),
        ("delta_i1_minus_past_self", "exploratory_relative_advantage"),
    ]
    derived_rows: list[dict[str, object]] = []
    for metric, role in derived_specs:
        stats = paired_summary(
            last[metric],
            label=f"posthoc:last-layer:{metric}",
            seed=args.seed,
            bootstrap_resamples=args.bootstrap_resamples,
            sign_flip_resamples=args.sign_flip_resamples,
        )
        derived_rows.append(
            {
                "transform": ROLL,
                "layer": max_layer,
                "metric": metric,
                "analysis_role": role,
                **stats,
            }
        )
    derived_path = output_dir / "posthoc_last_layer_paired_stats.csv"
    pd.DataFrame(derived_rows).to_csv(derived_path, index=False)

    correlation_rows: list[dict[str, object]] = []
    for left, right in [
        ("delta_mean_image1_mass_raw", "delta_i2_past_self_mass_raw"),
        ("delta_mean_image1_mass_raw", "delta_mean_text_mass_raw"),
        ("delta_i2_past_self_mass_raw", "delta_mean_text_mass_raw"),
        ("delta_mean_image1_mass_raw", "delta_row_entropy"),
    ]:
        left_rank = last[left].rank(method="average")
        right_rank = last[right].rank(method="average")
        correlation_rows.append(
            {
                "layer": max_layer,
                "left": left,
                "right": right,
                "pearson": float(last[left].corr(last[right], method="pearson")),
                "spearman": float(left_rank.corr(right_rank, method="pearson")),
                "analysis_role": "exploratory_correlation",
            }
        )
    correlation_path = output_dir / "posthoc_last_layer_correlations.csv"
    pd.DataFrame(correlation_rows).to_csv(correlation_path, index=False)

    i1_delta = last["delta_mean_image1_mass_raw"]
    self_delta = last["delta_i2_past_self_mass_raw"]
    sign_specs = [
        ("decrease", lambda values: values < 0),
        ("unchanged", lambda values: values == 0),
        ("increase", lambda values: values > 0),
    ]
    quadrant_specs = [
        (f"i1_{i1_name}_self_{self_name}", i1_mask(i1_delta) & self_mask(self_delta))
        for i1_name, i1_mask in sign_specs
        for self_name, self_mask in sign_specs
    ]
    quadrant_rows = [
        {
            "layer": max_layer,
            "quadrant": name,
            "case_count": int(mask.sum()),
            "case_fraction": float(mask.mean()),
            "analysis_role": "exploratory_joint_sign",
        }
        for name, mask in quadrant_specs
    ]
    quadrant_path = output_dir / "posthoc_last_layer_quadrants.csv"
    pd.DataFrame(quadrant_rows).to_csv(quadrant_path, index=False)

    layerwise_rows: list[dict[str, object]] = []
    layerwise_metrics = [
        "delta_mean_image1_mass_raw",
        "delta_i2_past_self_mass_raw",
        "delta_mean_text_mass_raw",
        "delta_i1_visual_history_share",
        "source_exact_minus_baseline_slot_exact",
    ]
    for layer, rows in roll.groupby("layer", sort=True):
        for metric in layerwise_metrics:
            stats = paired_summary(
                rows[metric],
                label=f"posthoc:layerwise:{layer}:{metric}",
                seed=args.seed,
                bootstrap_resamples=args.bootstrap_resamples,
                sign_flip_resamples=args.sign_flip_resamples,
                test_zero=False,
            )
            layerwise_rows.append(
                {
                    "transform": ROLL,
                    "layer": int(layer),
                    "metric": metric,
                    "analysis_role": "exploratory_layerwise_not_pooled",
                    **stats,
                }
            )
    layerwise_path = output_dir / "posthoc_layerwise_descriptive.csv"
    pd.DataFrame(layerwise_rows).to_csv(layerwise_path, index=False)

    residual = last["mass_accounting_residual"].to_numpy(dtype=np.float64)
    summary = {
        "input_csv": str(input_path),
        "input_sha256": input_sha256,
        "prompt_count": int(last["case_id"].nunique()),
        "primary_layer": max_layer,
        "analysis_is_exploratory": True,
        "inference_config": {
            "seed": int(args.seed),
            "bootstrap_resamples": int(args.bootstrap_resamples),
            "sign_flip_resamples": int(args.sign_flip_resamples),
            "derived_zero_test_family_size": len(derived_specs),
            "multiplicity_adjustment": "none_exploratory",
            "layerwise_zero_tests": False,
        },
        "no_equivalence_claim_without_prespecified_bound": True,
        "layers_are_not_pooled_as_independent_samples": True,
        "mass_accounting": {
            "mean_component_delta_sum": float(last["component_mass_delta_sum"].mean()),
            "mean_total_mass_delta": float(last["delta_mass_total_mean"].mean()),
            "max_abs_residual": float(np.abs(residual).max()),
        },
        "raw_means": {
            "baseline_slot_exact": float(last["exact_position_mass__baseline"].mean()),
            "roll_source_exact": float(last["source_index_exact_mass"].mean()),
            "roll_fixed_slot_exact": float(last["exact_position_mass"].mean()),
        },
        "behavioral_association_available": False,
        "artifacts": {
            "last_layer_paired_stats": str(derived_path),
            "last_layer_correlations": str(correlation_path),
            "last_layer_quadrants": str(quadrant_path),
            "layerwise_descriptive": str(layerwise_path),
        },
    }
    summary_path = output_dir / "posthoc_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
