#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_COLUMNS = [
    "mean_image1_mass_raw",
    "mean_text_mass_raw",
    "mean_image2_mass_raw",
    "position_band_mass",
    "content_band_mass",
    "expected_position_distance",
    "expected_content_distance",
    "row_entropy",
    "i2_total_self_mass_raw",
    "i2_past_self_mass_raw",
    "i2_diag_self_mass_raw",
    "i2_local_self_mass_raw",
    "i2_local_self_ratio",
    "target_mass_norm_all_queries",
    "target_mass_norm_target_queries",
    "target_mass_norm_content_shifted_target_queries",
    "distractor_mass_norm_all_queries",
    "target_minus_distractor_mass",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Qwen2.5-VL image2-shift flow probe summaries.")
    parser.add_argument("--input-dir", action="append", default=[], help="A probe output dir containing summary.json.")
    parser.add_argument("--input-dirs", nargs="*", default=[], help="Probe output dirs containing summary.json.")
    parser.add_argument("--output-dir", required=True)
    return parser


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def load_summary(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def flatten_rows(input_dir: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in summary.get("cases", []):
        for transform, transform_payload in case.get("transforms", {}).items():
            shift = (transform_payload.get("transform_record") or {}).get("shift") or {}
            for layer_payload in transform_payload.get("layers", []):
                row: dict[str, Any] = {
                    "input_dir": str(input_dir),
                    "case_id": case.get("case_id"),
                    "base_id": case.get("base_id", ""),
                    "question_id": case.get("question_id", ""),
                    "group": case.get("group", ""),
                    "source_dataset": case.get("source_dataset"),
                    "source_index": case.get("source_index"),
                    "image": case.get("image", ""),
                    "image_size": json.dumps(case.get("image_size"), ensure_ascii=False),
                    "target_box_xyxy": json.dumps(case.get("target_box_xyxy"), ensure_ascii=False),
                    "distractor_box_xyxy": json.dumps(case.get("distractor_box_xyxy"), ensure_ascii=False),
                    "mode": case.get("mode"),
                    "policy": case.get("policy"),
                    "transform": transform,
                    "layer": int(layer_payload["layer"]),
                    "npz_path": str(input_dir / layer_payload["npz_path"]),
                    "processed_shift_pixels": finite_float(shift.get("processed_shift_pixels", 0.0)),
                    "dx": finite_float(shift.get("dx", 0.0)),
                    "dy": finite_float(shift.get("dy", 0.0)),
                    "semantic_unit": shift.get("semantic_unit", ""),
                    "pixel_shift_kind": shift.get("pixel_shift_kind", ""),
                }
                for metric in METRIC_COLUMNS:
                    row[metric] = finite_float(layer_payload.get(metric))
                row["target_key_token_count"] = len(layer_payload.get("target_key_token_indices", []))
                row["target_query_token_count"] = len(layer_payload.get("target_query_token_indices", []))
                row["content_shifted_target_query_token_count"] = len(
                    layer_payload.get("content_shifted_target_query_token_indices", [])
                )
                rows.append(row)
    return rows


def add_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["transform"] == "baseline"][
        ["case_id", "layer", *METRIC_COLUMNS]
    ].rename(columns={metric: f"{metric}__baseline" for metric in METRIC_COLUMNS})
    merged = df.merge(baseline, on=["case_id", "layer"], how="left")
    for metric in METRIC_COLUMNS:
        merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}__baseline"]
    return merged


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("No rows.\n", encoding="utf-8")
        return
    view_cols = [
        "transform",
        "layer",
        "case_count",
        "delta_mean_image1_mass_raw",
        "delta_mean_image2_mass_raw",
        "delta_position_band_mass",
        "delta_content_band_mass",
        "delta_i2_total_self_mass_raw",
        "delta_i2_past_self_mass_raw",
        "delta_i2_local_self_ratio",
        "delta_row_entropy",
        "delta_target_mass_norm_all_queries",
    ]
    available = [col for col in view_cols if col in df.columns]
    lines = ["| " + " | ".join(available) + " |", "| " + " | ".join(["---"] * len(available)) + " |"]
    for _, row in df[available].iterrows():
        cells = []
        for col in available:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    input_dirs = [Path(p) for p in [*args.input_dir, *args.input_dirs]]
    if not input_dirs:
        raise ValueError("Provide at least one --input-dir or --input-dirs entry.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for input_dir in input_dirs:
        summary = load_summary(input_dir)
        run_summaries.append(
            {
                "input_dir": str(input_dir),
                "manifest": summary.get("manifest"),
                "model_path": summary.get("model_path"),
                "case_count": summary.get("case_count"),
                "mode": summary.get("mode"),
                "attn_layers": summary.get("attn_layers"),
                "selected_layers": summary.get("selected_layers"),
                "transforms": summary.get("transforms"),
            }
        )
        rows.extend(flatten_rows(input_dir, summary))

    case_layer = pd.DataFrame(rows)
    if case_layer.empty:
        raise RuntimeError("No layer rows found in summaries.")
    case_layer.to_csv(output_dir / "case_layer_metrics.csv", index=False)

    with_delta = add_delta_columns(case_layer)
    with_delta.to_csv(output_dir / "case_layer_metrics_with_delta.csv", index=False)

    metric_and_delta_cols = [
        col for col in with_delta.columns if col in METRIC_COLUMNS or col.startswith("delta_")
    ]
    aggregate = (
        with_delta.groupby(["transform", "layer"], as_index=False)
        .agg(
            case_count=("case_id", "nunique"),
            **{col: (col, "mean") for col in metric_and_delta_cols},
        )
        .sort_values(["transform", "layer"])
    )
    aggregate.to_csv(output_dir / "aggregate_layer_metrics.csv", index=False)

    max_layer = int(with_delta["layer"].max())
    last_layer = aggregate[aggregate["layer"] == max_layer].copy()
    last_layer.to_csv(output_dir / "last_layer_metrics.csv", index=False)

    delta_rows = with_delta[with_delta["transform"] != "baseline"].copy()
    delta_summary = (
        delta_rows.groupby(["transform"], as_index=False)
        .agg(
            case_count=("case_id", "nunique"),
            **{f"{col}_mean": (col, "mean") for col in delta_rows.columns if col.startswith("delta_")},
        )
        .sort_values("transform")
    )
    delta_summary.to_csv(output_dir / "delta_vs_baseline_all_layers.csv", index=False)

    last_delta = with_delta[(with_delta["transform"] != "baseline") & (with_delta["layer"] == max_layer)].copy()
    last_delta_summary = (
        last_delta.groupby(["transform"], as_index=False)
        .agg(
            case_count=("case_id", "nunique"),
            **{f"{col}_mean": (col, "mean") for col in last_delta.columns if col.startswith("delta_")},
        )
        .sort_values("transform")
    )
    last_delta_summary.to_csv(output_dir / "delta_vs_baseline_last_layer.csv", index=False)
    write_markdown_table(last_layer, output_dir / "last_layer_metrics.md")

    analysis_summary = {
        "input_dirs": [str(p) for p in input_dirs],
        "case_count": int(case_layer["case_id"].nunique()),
        "row_count": int(len(case_layer)),
        "layers": [int(v) for v in sorted(case_layer["layer"].unique().tolist())],
        "transforms": sorted(case_layer["transform"].unique().tolist()),
        "max_layer": max_layer,
        "run_summaries": run_summaries,
        "artifacts": {
            "case_layer_metrics_csv": str(output_dir / "case_layer_metrics.csv"),
            "case_layer_metrics_with_delta_csv": str(output_dir / "case_layer_metrics_with_delta.csv"),
            "aggregate_layer_metrics_csv": str(output_dir / "aggregate_layer_metrics.csv"),
            "last_layer_metrics_csv": str(output_dir / "last_layer_metrics.csv"),
            "delta_vs_baseline_all_layers_csv": str(output_dir / "delta_vs_baseline_all_layers.csv"),
            "delta_vs_baseline_last_layer_csv": str(output_dir / "delta_vs_baseline_last_layer.csv"),
            "last_layer_metrics_md": str(output_dir / "last_layer_metrics.md"),
        },
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(analysis_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(analysis_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
