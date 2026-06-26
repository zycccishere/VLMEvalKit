#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate IQIQ prefill-attention visualization artifacts.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mass-sum-tol", type=float, default=5e-3)
    parser.add_argument("--matrix-mass-tol", type=float, default=2e-3)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--min-figures", type=int, default=1)
    parser.add_argument(
        "--require-matrix",
        action="append",
        default=["i2_to_i1", "q2_to_q1"],
        help="Matrix spec that must appear at least once. Can be passed multiple times.",
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_mass_sums(df: pd.DataFrame, tol: float) -> list[dict[str, Any]]:
    failures = []
    grouped = df.groupby(["sample_index", "layer", "head", "query_group"], as_index=False)["mass"].sum()
    for _, row in grouped.iterrows():
        total = float(row["mass"])
        if not np.isfinite(total) or abs(total - 1.0) > tol:
            failures.append(
                {
                    "check": "mass_sum",
                    "sample_index": int(row["sample_index"]),
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    "query_group": str(row["query_group"]),
                    "mass_sum": total,
                    "tolerance": float(tol),
                }
            )
    return failures


def validate_matrix_masses(output_dir: Path, df: pd.DataFrame, manifest: pd.DataFrame, tol: float) -> list[dict[str, Any]]:
    failures = []
    for _, item in manifest.iterrows():
        path = output_dir / str(item["path"])
        if not path.exists():
            failures.append({"check": "matrix_file_exists", "path": str(path)})
            continue
        with np.load(path) as npz:
            head_mass = np.asarray(npz["head_mass"], dtype=np.float64)
            matrix_stored = bool(int(npz["matrix_stored"][0]))
            query_mean = np.asarray(npz["query_mean"], dtype=np.float64)
            if query_mean.ndim != 2:
                failures.append(
                    {
                        "check": "query_mean_rank",
                        "path": str(path),
                        "shape": list(query_mean.shape),
                    }
                )
                continue
            derived_head_mass = query_mean.sum(axis=-1)
            local_delta = np.max(np.abs(derived_head_mass - head_mass)) if head_mass.size else np.nan
            if not np.isfinite(local_delta) or local_delta > tol:
                failures.append(
                    {
                        "check": "npz_head_mass_matches_query_mean",
                        "path": str(path),
                        "max_delta": float(local_delta),
                        "tolerance": float(tol),
                    }
                )
            if matrix_stored and "matrix" in npz:
                matrix = np.asarray(npz["matrix"], dtype=np.float64)
                if matrix.ndim != 3:
                    failures.append(
                        {
                            "check": "matrix_rank",
                            "path": str(path),
                            "shape": list(matrix.shape),
                        }
                    )
                elif matrix.shape[0] != query_mean.shape[0] or matrix.shape[2] != query_mean.shape[1]:
                    failures.append(
                        {
                            "check": "matrix_shape_matches_query_mean",
                            "path": str(path),
                            "matrix_shape": list(matrix.shape),
                            "query_mean_shape": list(query_mean.shape),
                        }
                    )

        sample_index = int(item["sample_index"])
        layer = int(item["layer"])
        query_group = str(item["query_group"])
        key_group = str(item["key_group"])
        rows = df[
            (df["sample_index"] == sample_index)
            & (df["layer"] == layer)
            & (df["query_group"] == query_group)
            & (df["key_group"] == key_group)
        ].sort_values("head")
        if rows.empty:
            failures.append(
                {
                    "check": "csv_group_mass_exists",
                    "sample_index": sample_index,
                    "layer": layer,
                    "query_group": query_group,
                    "key_group": key_group,
                }
            )
            continue
        csv_mass = rows["mass"].to_numpy(dtype=np.float64)
        if csv_mass.shape != head_mass.shape:
            failures.append(
                {
                    "check": "csv_npz_head_count",
                    "path": str(path),
                    "csv_shape": list(csv_mass.shape),
                    "npz_shape": list(head_mass.shape),
                }
            )
            continue
        delta = float(np.max(np.abs(csv_mass - head_mass)))
        if delta > tol:
            failures.append(
                {
                    "check": "csv_mass_matches_npz_head_mass",
                    "path": str(path),
                    "max_delta": delta,
                    "tolerance": float(tol),
                }
            )
    return failures


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    group_path = output_dir / "group_masses.csv"
    manifest_path = output_dir / "matrix_manifest.csv"
    figures = sorted((output_dir / "figures").glob("*.png"))

    failures: list[dict[str, Any]] = []
    for required in [summary_path, group_path, manifest_path]:
        if not required.exists():
            failures.append({"check": "required_file_exists", "path": str(required)})

    if failures:
        report = {"ok": False, "failures": failures}
        (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    summary = load_json(summary_path)
    group_df = pd.read_csv(group_path)
    manifest = pd.read_csv(manifest_path)

    if int(summary.get("sample_count", 0)) < args.min_samples:
        failures.append(
            {
                "check": "min_samples",
                "sample_count": int(summary.get("sample_count", 0)),
                "required": int(args.min_samples),
            }
        )
    if len(figures) < args.min_figures:
        failures.append(
            {
                "check": "min_figures",
                "figure_count": len(figures),
                "required": int(args.min_figures),
            }
        )
    for name in args.require_matrix:
        if manifest.empty or name not in set(manifest["name"].astype(str).tolist()):
            failures.append({"check": "required_matrix_present", "name": name})

    failures.extend(validate_mass_sums(group_df, args.mass_sum_tol))
    failures.extend(validate_matrix_masses(output_dir, group_df, manifest, args.matrix_mass_tol))

    report = {
        "ok": not failures,
        "output_dir": str(output_dir),
        "sample_count": int(summary.get("sample_count", 0)),
        "layers": summary.get("selected_layers", []),
        "group_mass_rows": int(len(group_df)),
        "matrix_count": int(len(manifest)),
        "figure_count": int(len(figures)),
        "failures": failures,
    }
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
