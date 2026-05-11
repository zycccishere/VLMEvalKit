#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize layered decode-attention outputs from standard vs rope-align probe runs."
    )
    parser.add_argument("--standard-dir", required=True)
    parser.add_argument("--rope-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def load_decode(path: Path, condition: str) -> pd.DataFrame:
    df = pd.read_csv(path / "decode_attention.csv")
    if df.empty:
        return df
    if {"sample_index", "layer"}.issubset(df.columns):
        # Older probe outputs flattened multi-layer decode records and wrote a
        # global event index into `step`, which stretches the x-axis by the
        # number of traced layers. Recover the true per-layer decode step here.
        df = df.sort_values(["sample_index", "layer", "step"]).copy()
        df["decode_step"] = df.groupby(["sample_index", "layer"]).cumcount()
    else:
        df = df.copy()
        df["decode_step"] = df["step"]
    grouped = (
        df.groupby(["layer", "decode_step"], as_index=False)[["image1_mass", "image2_mass", "text_mass"]]
        .mean()
        .sort_values(["layer", "decode_step"])
        .rename(columns={"decode_step": "step"})
    )
    grouped["condition"] = condition
    grouped["image2_over_image1_ratio"] = grouped["image2_mass"] / grouped["image1_mass"].clip(lower=1e-8)
    grouped["text_share"] = grouped["text_mass"] / (
        grouped["image1_mass"] + grouped["image2_mass"] + grouped["text_mass"]
    ).clip(lower=1e-8)
    return grouped


def plot_ratio(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for layer in sorted(df["layer"].unique()):
        layer_df = df[df["layer"] == layer]
        ax.plot(layer_df["step"], layer_df["image2_over_image1_ratio"], label=f"L{layer}")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("decode step")
    ax.set_ylabel("image2 / image1 mass ratio")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ratio_compare(standard: pd.DataFrame, rope: pd.DataFrame, out_path: Path) -> None:
    if standard.empty and rope.empty:
        return
    layers = sorted(set(standard.get("layer", pd.Series(dtype=int)).dropna().astype(int).tolist()) |
                    set(rope.get("layer", pd.Series(dtype=int)).dropna().astype(int).tolist()))
    if not layers:
        return
    fig, axes = plt.subplots(len(layers), 1, figsize=(12, 3.2 * len(layers)), sharex=True)
    if len(layers) == 1:
        axes = [axes]
    for ax, layer in zip(axes, layers):
        std_df = standard[standard["layer"] == layer]
        rope_df = rope[rope["layer"] == layer]
        if not std_df.empty:
            ax.plot(
                std_df["step"],
                std_df["image2_over_image1_ratio"],
                label="standard",
                color="#4c72b0",
                linewidth=1.8,
            )
        if not rope_df.empty:
            ax.plot(
                rope_df["step"],
                rope_df["image2_over_image1_ratio"],
                label="rope_align",
                color="#dd8452",
                linewidth=1.8,
            )
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_ylabel(f"L{layer}\nratio")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("decode step")
    fig.suptitle("Decode image2/image1 Mass Ratio: standard vs rope_align by layer", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_component_compare(standard: pd.DataFrame, rope: pd.DataFrame, out_path: Path) -> None:
    if standard.empty and rope.empty:
        return
    layers = sorted(set(standard.get("layer", pd.Series(dtype=int)).dropna().astype(int).tolist()) |
                    set(rope.get("layer", pd.Series(dtype=int)).dropna().astype(int).tolist()))
    if not layers:
        return
    fig, axes = plt.subplots(len(layers), 2, figsize=(14, 3.2 * len(layers)), sharex=True, sharey=False)
    if len(layers) == 1:
        axes = [axes]
    component_specs = [
        ("image1_mass", "image1", "#4c72b0"),
        ("image2_mass", "image2", "#dd8452"),
        ("text_mass", "text", "#55a868"),
    ]
    for row_idx, layer in enumerate(layers):
        for col_idx, (name, df, title) in enumerate(
            [("standard", standard, "standard"), ("rope_align", rope, "rope_align")]
        ):
            ax = axes[row_idx][col_idx]
            layer_df = df[df["layer"] == layer]
            if not layer_df.empty:
                for key, label, color in component_specs:
                    ax.plot(layer_df["step"], layer_df[key], label=label, color=color, linewidth=1.6)
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.set_ylabel(f"L{layer}\nmass")
            if row_idx == len(layers) - 1:
                ax.set_xlabel("decode step")
            if row_idx == 0 and col_idx == 1:
                ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Decode attention mass trajectories by layer and condition", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_delta_components(merged: pd.DataFrame, out_path: Path) -> None:
    if merged.empty:
        return
    layers = sorted(merged["layer"].dropna().astype(int).unique().tolist())
    if not layers:
        return
    fig, axes = plt.subplots(len(layers), 1, figsize=(12, 3.2 * len(layers)), sharex=True)
    if len(layers) == 1:
        axes = [axes]
    component_specs = [
        ("image1_mass_delta", "delta image1", "#4c72b0"),
        ("image2_mass_delta", "delta image2", "#dd8452"),
        ("text_mass_delta", "delta text", "#55a868"),
    ]
    for ax, layer in zip(axes, layers):
        layer_df = merged[merged["layer"] == layer]
        for key, label, color in component_specs:
            ax.plot(layer_df["step"], layer_df[key], label=label, color=color, linewidth=1.6)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_ylabel(f"L{layer}\ndelta")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("decode step")
    fig.suptitle("Decode attention mass deltas: rope_align - standard", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    standard = load_decode(Path(args.standard_dir), "standard")
    rope = load_decode(Path(args.rope_dir), "rope_align")

    standard.to_csv(output_dir / "standard_decode_layer_step_summary.csv", index=False)
    rope.to_csv(output_dir / "rope_decode_layer_step_summary.csv", index=False)

    merged = standard.merge(
        rope,
        on=["layer", "step"],
        suffixes=("_standard", "_rope"),
        how="outer",
    ).sort_values(["layer", "step"])
    for key in ["image1_mass", "image2_mass", "text_mass", "image2_over_image1_ratio", "text_share"]:
        merged[f"{key}_delta"] = merged[f"{key}_rope"] - merged[f"{key}_standard"]
    merged.to_csv(output_dir / "rope_minus_standard_decode_layer_step_summary.csv", index=False)

    plot_ratio(standard, output_dir / "standard_decode_ratio_by_layer.png", "Standard decode ratio by layer")
    plot_ratio(rope, output_dir / "rope_decode_ratio_by_layer.png", "Rope-align decode ratio by layer")
    plot_ratio_compare(standard, rope, output_dir / "compare_decode_ratio_by_layer.png")
    plot_component_compare(standard, rope, output_dir / "compare_decode_component_mass_by_layer.png")
    plot_delta_components(merged, output_dir / "compare_decode_component_delta_by_layer.png")

    report = {
        "standard_dir": str(Path(args.standard_dir)),
        "rope_dir": str(Path(args.rope_dir)),
        "standard_layers": sorted(standard["layer"].dropna().astype(int).unique().tolist()) if not standard.empty else [],
        "rope_layers": sorted(rope["layer"].dropna().astype(int).unique().tolist()) if not rope.empty else [],
        "standard_mean_ratio_by_layer": (
            standard.groupby("layer")["image2_over_image1_ratio"].mean().to_dict() if not standard.empty else {}
        ),
        "rope_mean_ratio_by_layer": (
            rope.groupby("layer")["image2_over_image1_ratio"].mean().to_dict() if not rope.empty else {}
        ),
        "generated_plots": [
            "standard_decode_ratio_by_layer.png",
            "rope_decode_ratio_by_layer.png",
            "compare_decode_ratio_by_layer.png",
            "compare_decode_component_mass_by_layer.png",
            "compare_decode_component_delta_by_layer.png",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
