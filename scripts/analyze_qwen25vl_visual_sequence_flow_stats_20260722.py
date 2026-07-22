#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_TRANSFORMS = [
    "baseline",
    "shift_right_half_vit_token",
    "shift_right_one_vit_token",
    "shift_right_one_llm_token",
    "visual_sequence_roll_right_1",
]

PRIMARY_DELTA_METRICS = [
    "delta_mean_image1_mass_raw",
    "delta_mean_text_mass_raw",
    "delta_mean_image2_mass_raw",
    "delta_i2_past_self_mass_raw",
    "delta_exact_position_mass",
    "delta_position_band_mass",
    "delta_row_entropy",
]

SEQUENCE_SOURCE_METRICS = [
    "source_minus_position_exact_mass",
    "source_index_exact_mass",
    "source_index_band_mass",
    "expected_source_index_distance",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired prompt-level inference for the Qwen visual-sequence flow experiment."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-cases", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--sign-flip-resamples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--strict-canonical", action="store_true")
    return parser


def derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def paired_summary(
    values: pd.Series,
    *,
    label: str,
    seed: int,
    bootstrap_resamples: int,
    sign_flip_resamples: int,
    test_zero: bool = True,
) -> dict[str, float | int]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(array)
    if not bool(finite.all()):
        raise ValueError(
            f"Non-finite prompt-level values for {label}: "
            f"finite={int(finite.sum())} total={int(array.size)}"
        )
    if array.size == 0:
        raise ValueError(f"No prompt-level values for {label}")

    bootstrap_rng = np.random.default_rng(derived_seed(seed, f"{label}:bootstrap"))
    sign_flip_rng = np.random.default_rng(derived_seed(seed, f"{label}:sign-flip"))
    sample_count = int(array.size)
    observed_mean = float(array.mean())

    bootstrap_means = np.empty(bootstrap_resamples, dtype=np.float64)
    chunk_size = 2_000
    offset = 0
    while offset < bootstrap_resamples:
        count = min(chunk_size, bootstrap_resamples - offset)
        indices = bootstrap_rng.integers(0, sample_count, size=(count, sample_count))
        bootstrap_means[offset : offset + count] = array[indices].mean(axis=1)
        offset += count
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])

    p_two_sided = float("nan")
    if test_zero:
        extreme_count = 0
        offset = 0
        threshold = abs(observed_mean) - 1e-15
        while offset < sign_flip_resamples:
            count = min(10_000, sign_flip_resamples - offset)
            signs = sign_flip_rng.integers(0, 2, size=(count, sample_count), dtype=np.int8)
            signs = signs * 2 - 1
            null_means = (signs @ array) / sample_count
            extreme_count += int(np.count_nonzero(np.abs(null_means) >= threshold))
            offset += count
        p_two_sided = (extreme_count + 1.0) / (sign_flip_resamples + 1.0)

    return {
        "prompt_count": sample_count,
        "mean": observed_mean,
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if sample_count > 1 else 0.0,
        "positive_fraction": float(np.mean(array > 0)),
        "negative_fraction": float(np.mean(array < 0)),
        "zero_fraction": float(np.mean(array == 0)),
        "bootstrap_ci95_low": float(ci_low),
        "bootstrap_ci95_high": float(ci_high),
        "zero_null_tested": bool(test_zero),
        "sign_flip_p_two_sided": float(p_two_sided),
    }


def validate_contract(df: pd.DataFrame, *, expected_cases: int, strict_canonical: bool) -> None:
    required = {"case_id", "transform", "layer", *PRIMARY_DELTA_METRICS, *SEQUENCE_SOURCE_METRICS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required analysis columns: {missing}")
    duplicate = df.duplicated(["case_id", "transform", "layer"], keep=False)
    if duplicate.any():
        rows = df.loc[duplicate, ["case_id", "transform", "layer"]].head(10).to_dict("records")
        raise ValueError(f"Duplicate case/transform/layer rows: {rows}")

    transforms = sorted(df["transform"].dropna().astype(str).unique().tolist())
    if strict_canonical and transforms != sorted(CANONICAL_TRANSFORMS):
        raise ValueError(f"Unexpected transform set: {transforms}")
    layers = sorted(int(item) for item in df["layer"].unique().tolist())
    if strict_canonical and layers != [60, 61, 62, 63]:
        raise ValueError(f"Strict Qwen32 run requires layers [60, 61, 62, 63], got {layers}")
    if strict_canonical and expected_cases <= 0:
        raise ValueError("Strict canonical analysis requires an explicit positive --expected-cases")
    if expected_cases:
        observed = int(df["case_id"].nunique())
        if observed != expected_cases:
            raise ValueError(f"Expected {expected_cases} unique cases, got {observed}")
        expected_cells = pd.MultiIndex.from_product(
            [transforms, layers],
            names=["transform", "layer"],
        )
        counts = (
            df.groupby(["transform", "layer"])["case_id"]
            .nunique()
            .reindex(expected_cells, fill_value=0)
        )
        if not bool((counts == expected_cases).all()):
            raise ValueError(f"Incomplete transform/layer cells: {counts[counts != expected_cases].to_dict()}")

    nonbaseline = df[df["transform"] != "baseline"]
    finite_contracts = [
        (nonbaseline, PRIMARY_DELTA_METRICS, "non-baseline primary deltas"),
        (
            df[df["transform"] == "visual_sequence_roll_right_1"],
            SEQUENCE_SOURCE_METRICS,
            "visual-sequence source metrics",
        ),
    ]
    for frame, metrics, label in finite_contracts:
        for metric in metrics:
            values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=np.float64)
            finite = np.isfinite(values)
            if not bool(finite.all()):
                bad_rows = frame.loc[
                    ~finite,
                    ["case_id", "transform", "layer"],
                ].head(10).to_dict("records")
                raise ValueError(
                    f"Non-finite {label} in {metric}: "
                    f"finite={int(finite.sum())} total={int(values.size)} rows={bad_rows}"
                )


def main() -> int:
    args = build_parser().parse_args()
    if args.bootstrap_resamples <= 0 or args.sign_flip_resamples <= 0:
        raise ValueError("Resample counts must be positive")

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    validate_contract(
        df,
        expected_cases=args.expected_cases,
        strict_canonical=args.strict_canonical,
    )

    max_layer = int(df["layer"].max())
    last_layer = df[(df["layer"] == max_layer) & (df["transform"] != "baseline")].copy()
    paired_rows: list[dict[str, object]] = []
    for transform, transform_rows in last_layer.groupby("transform", sort=True):
        for metric in PRIMARY_DELTA_METRICS:
            is_primary = transform == "visual_sequence_roll_right_1" and metric in {
                "delta_mean_image1_mass_raw",
                "delta_i2_past_self_mass_raw",
            }
            expected_direction = {
                "delta_mean_image1_mass_raw": "increase",
                "delta_i2_past_self_mass_raw": "decrease",
            }.get(metric, "descriptive") if is_primary else "descriptive"
            stats = paired_summary(
                transform_rows[metric],
                label=f"last-layer:{transform}:{metric}",
                seed=args.seed,
                bootstrap_resamples=args.bootstrap_resamples,
                sign_flip_resamples=args.sign_flip_resamples,
            )
            paired_rows.append(
                {
                    "transform": transform,
                    "layer": max_layer,
                    "metric": metric,
                    "comparison": "paired_delta_vs_baseline",
                    "analysis_role": "primary_hypothesis" if is_primary else "secondary_diagnostic",
                    "expected_direction": expected_direction,
                    **stats,
                }
            )
    paired_df = pd.DataFrame(paired_rows)
    paired_path = output_dir / "paired_last_layer_stats.csv"
    paired_df.to_csv(paired_path, index=False)

    sequence_rows = last_layer[last_layer["transform"] == "visual_sequence_roll_right_1"]
    source_rows: list[dict[str, object]] = []
    for metric in SEQUENCE_SOURCE_METRICS:
        stats = paired_summary(
            sequence_rows[metric],
            label=f"sequence-source:{metric}",
            seed=args.seed,
            bootstrap_resamples=args.bootstrap_resamples,
            sign_flip_resamples=args.sign_flip_resamples,
            test_zero=metric == "source_minus_position_exact_mass",
        )
        source_rows.append(
            {
                "transform": "visual_sequence_roll_right_1",
                "layer": max_layer,
                "metric": metric,
                "comparison": (
                    "within_condition_source_minus_position"
                    if metric == "source_minus_position_exact_mass"
                    else "within_condition_descriptive"
                ),
                **stats,
            }
        )
    source_df = pd.DataFrame(source_rows)
    source_path = output_dir / "sequence_source_alignment_stats.csv"
    source_df.to_csv(source_path, index=False)

    robustness_rows: list[dict[str, object]] = []
    nonbaseline = df[df["transform"] != "baseline"]
    for (transform, layer), rows in nonbaseline.groupby(["transform", "layer"], sort=True):
        for metric in PRIMARY_DELTA_METRICS:
            values = pd.to_numeric(rows[metric], errors="coerce")
            values = values[np.isfinite(values)]
            robustness_rows.append(
                {
                    "transform": transform,
                    "layer": int(layer),
                    "metric": metric,
                    "prompt_count": int(values.size),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                }
            )
    robustness_path = output_dir / "layerwise_robustness_descriptive.csv"
    pd.DataFrame(robustness_rows).to_csv(robustness_path, index=False)

    summary = {
        "input_csv": str(input_csv),
        "max_layer_primary": max_layer,
        "layers_are_not_pooled_as_independent_samples": True,
        "prompt_count": int(df["case_id"].nunique()),
        "transforms": sorted(df["transform"].unique().tolist()),
        "bootstrap_resamples": args.bootstrap_resamples,
        "sign_flip_resamples": args.sign_flip_resamples,
        "seed": args.seed,
        "artifacts": {
            "paired_last_layer_stats": str(paired_path),
            "sequence_source_alignment_stats": str(source_path),
            "layerwise_robustness_descriptive": str(robustness_path),
        },
    }
    summary_path = output_dir / "paired_stats_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
