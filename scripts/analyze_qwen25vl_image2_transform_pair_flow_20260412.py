#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze baseline-vs-shift transform-pair flow outputs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-8
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(float(p.sum()), eps)
    q = q / max(float(q.sum()), eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(np.where(p > 0, p * np.log((p + eps) / (m + eps)), 0.0))
    kl_qm = np.sum(np.where(q > 0, q * np.log((q + eps) / (m + eps)), 0.0))
    return float(0.5 * (kl_pm + kl_qm))


def load_matrix(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    return np.asarray(data["matrix_norm"], dtype=np.float32)


def resize_matrix(mat: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if tuple(mat.shape) == tuple(target_shape):
        return mat.astype(np.float32, copy=False)
    tensor = torch.from_numpy(mat.astype(np.float32, copy=False))[None, None, :, :]
    resized = F.interpolate(tensor, size=target_shape, mode="bilinear", align_corners=False)
    return resized[0, 0].cpu().numpy()


def pad_profile(profile: list[float], length: int) -> list[float]:
    if len(profile) >= length:
        return profile[:length]
    return profile + [float("nan")] * (length - len(profile))


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    shift_transform = None
    rows: list[dict[str, object]] = []
    aggregate_mats: dict[tuple[str, int], list[np.ndarray]] = {}
    matrix_shapes: dict[int, list[tuple[int, int]]] = {}

    for case in summary["cases"]:
        transform_names = list(case["transforms"].keys())
        baseline_name = "baseline"
        shift_name = next(name for name in transform_names if name != baseline_name)
        shift_transform = shift_name
        for transform_name, transform_payload in case["transforms"].items():
            for layer_info in transform_payload["layers"]:
                matrix = load_matrix(input_dir / layer_info["npz_path"])
                matrix_shapes.setdefault(int(layer_info["layer"]), []).append(tuple(int(v) for v in matrix.shape))
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "group": case.get("group", ""),
                        "source_dataset": case["source_dataset"],
                        "source_index": case["source_index"],
                        "transform": transform_name,
                        "layer": int(layer_info["layer"]),
                        "local_correspondence_band_mass": float(layer_info["local_correspondence_band_mass"]),
                        "expected_distance_from_diagonal": float(layer_info["expected_distance_from_diagonal"]),
                        "row_entropy": float(layer_info["row_entropy"]),
                        "mean_image1_mass_raw": float(layer_info["mean_image1_mass_raw"]),
                        "mean_text_mass_raw": float(layer_info["mean_text_mass_raw"]),
                        "mean_image2_mass_raw": float(layer_info["mean_image2_mass_raw"]),
                        "query_rows": int(matrix.shape[0]),
                        "key_cols": int(matrix.shape[1]),
                        "npz_path": layer_info["npz_path"],
                    }
                )
                aggregate_mats.setdefault((transform_name, int(layer_info["layer"])), []).append(matrix)

    case_df = pd.DataFrame(rows)
    case_df.to_csv(output_dir / "case_layer_metrics.csv", index=False)

    metric_cols = [
        "local_correspondence_band_mass",
        "expected_distance_from_diagonal",
        "row_entropy",
        "mean_image1_mass_raw",
        "mean_text_mass_raw",
        "mean_image2_mass_raw",
    ]
    agg_df = case_df.groupby(["transform", "layer"], as_index=False)[metric_cols].mean()

    js_rows: list[dict[str, object]] = []
    layers = sorted(case_df["layer"].unique().tolist())
    baseline_mats = {}
    shift_mats = {}
    target_shapes: dict[int, tuple[int, int]] = {}
    for layer in layers:
        shapes = matrix_shapes[layer]
        target_shape = (
            int(round(float(np.median([shape[0] for shape in shapes])))),
            int(round(float(np.median([shape[1] for shape in shapes])))),
        )
        target_shapes[layer] = target_shape
        baseline_resized = [resize_matrix(mat, target_shape) for mat in aggregate_mats[("baseline", layer)]]
        shift_resized = [resize_matrix(mat, target_shape) for mat in aggregate_mats[(shift_transform, layer)]]
        baseline_mean = np.stack(baseline_resized, axis=0).mean(axis=0)
        shift_mean = np.stack(shift_resized, axis=0).mean(axis=0)
        baseline_mats[layer] = baseline_mean
        shift_mats[layer] = shift_mean
        js_rows.append(
            {
                "layer": layer,
                "transform": shift_transform,
                "jsd_vs_baseline": js_divergence(baseline_mean.ravel(), shift_mean.ravel()),
            }
        )
    js_df = pd.DataFrame(js_rows)
    agg_df = agg_df.merge(js_df, on=["transform", "layer"], how="left")
    agg_df.to_csv(output_dir / "aggregate_layer_metrics.csv", index=False)

    max_profile_len = 0
    profile_records = []
    for case in summary["cases"]:
        for transform_name, transform_payload in case["transforms"].items():
            for layer_info in transform_payload["layers"]:
                max_profile_len = max(max_profile_len, len(layer_info["distance_profile"]))
                profile_records.append(
                    {
                        "case_id": case["case_id"],
                        "transform": transform_name,
                        "layer": int(layer_info["layer"]),
                        "distance_profile": list(layer_info["distance_profile"]),
                    }
                )

    profile_rows = []
    for record in profile_records:
        padded = pad_profile(record["distance_profile"], max_profile_len)
        row = {"case_id": record["case_id"], "transform": record["transform"], "layer": record["layer"]}
        for idx, value in enumerate(padded):
            row[f"d{idx}"] = value
        profile_rows.append(row)
    profile_df = pd.DataFrame(profile_rows)
    profile_mean = profile_df.groupby(["transform", "layer"], as_index=False).mean(numeric_only=True)
    profile_mean.to_csv(output_dir / "distance_profile_mean.csv", index=False)

    fig, axes = plt.subplots(len(layers), 3, figsize=(12, 3.2 * len(layers)))
    if len(layers) == 1:
        axes = np.asarray([axes])
    for row_idx, layer in enumerate(layers):
        baseline_mean = baseline_mats[layer]
        shift_mean = shift_mats[layer]
        delta = shift_mean - baseline_mean
        vmax = max(float(baseline_mean.max()), float(shift_mean.max()))
        # Use a robust symmetric color scale for the delta panel so a few extreme
        # entries do not flatten the visible contrast of the bulk pattern.
        delta_abs = np.abs(delta).ravel()
        dmax = max(abs(float(delta.min())), abs(float(delta.max())))
        drobust = float(np.percentile(delta_abs, 98))
        dscale = max(drobust, dmax * 0.35, 1e-6)
        for ax, mat, title, cmap, lim in [
            (axes[row_idx, 0], baseline_mean, f"L{layer} baseline", "viridis", (0.0, vmax)),
            (axes[row_idx, 1], shift_mean, f"L{layer} shift", "viridis", (0.0, vmax)),
            (axes[row_idx, 2], np.clip(delta, -dscale, dscale), f"L{layer} shift-baseline", "coolwarm", (-dscale, dscale)),
        ]:
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=lim[0], vmax=lim[1])
            ax.set_xlabel("image1 key token")
            ax.set_ylabel("image2 query token")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "layer_heatmaps.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, metric in zip(
        axes,
        [
            "local_correspondence_band_mass",
            "expected_distance_from_diagonal",
            "row_entropy",
            "jsd_vs_baseline",
        ],
    ):
        base = agg_df[agg_df["transform"] == "baseline"]
        shift = agg_df[agg_df["transform"] == shift_transform]
        ax.plot(base["layer"], base[metric], marker="o", label="baseline")
        ax.plot(shift["layer"], shift[metric], marker="o", label=shift_transform)
        ax.set_title(metric)
        ax.set_xlabel("layer")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "metric_curves.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(layers), figsize=(4.2 * len(layers), 4))
    if len(layers) == 1:
        axes = [axes]
    for ax, layer in zip(axes, layers):
        base = profile_mean[(profile_mean["transform"] == "baseline") & (profile_mean["layer"] == layer)]
        shift = profile_mean[(profile_mean["transform"] == shift_transform) & (profile_mean["layer"] == layer)]
        x = [int(col[1:]) for col in profile_mean.columns if col.startswith("d")]
        base_y = [float(base.iloc[0][f"d{i}"]) for i in x]
        shift_y = [float(shift.iloc[0][f"d{i}"]) for i in x]
        ax.plot(x, base_y, marker="o", label="baseline")
        ax.plot(x, shift_y, marker="o", label=shift_transform)
        ax.set_title(f"L{layer} distance profile")
        ax.set_xlabel("chebyshev distance from diagonal")
        ax.set_ylabel("mean row mass")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "distance_profiles.png", dpi=200)
    plt.close(fig)

    summary_out = {
        "input_dir": str(input_dir),
        "shift_transform": shift_transform,
        "case_count": int(case_df["case_id"].nunique()),
        "layers": layers,
        "target_shapes": {f"L{layer}": list(target_shapes[layer]) for layer in layers},
        "artifacts": {
            "case_layer_metrics_csv": str(output_dir / "case_layer_metrics.csv"),
            "aggregate_layer_metrics_csv": str(output_dir / "aggregate_layer_metrics.csv"),
            "distance_profile_mean_csv": str(output_dir / "distance_profile_mean.csv"),
            "layer_heatmaps_png": str(output_dir / "layer_heatmaps.png"),
            "metric_curves_png": str(output_dir / "metric_curves.png"),
            "distance_profiles_png": str(output_dir / "distance_profiles.png"),
        },
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
