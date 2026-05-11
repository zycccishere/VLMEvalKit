#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BIN_SPECS = [
    ("0-9", 0, 9),
    ("10-19", 10, 19),
    ("20-39", 20, 39),
    ("40-63", 40, 63),
    ("64-95", 64, 95),
    ("96-127", 96, 127),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot rope-only faithful image2-image1 gap curves for one dataset and two corruption settings."
    )
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--blank-analysis-logprob-dir", required=True)
    parser.add_argument("--blank-analysis-margin-dir", required=True)
    parser.add_argument("--blank-rope-dir", required=True)
    parser.add_argument("--swap-analysis-logprob-dir", required=True)
    parser.add_argument("--swap-analysis-margin-dir", required=True)
    parser.add_argument("--swap-rope-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-generated-tokens", type=int, default=64)
    return parser


def load_rope_mean(
    analysis_dir: Path,
    raw_rope_dir: Path,
    min_generated_tokens: int,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, float]]:
    gen = pd.read_csv(raw_rope_dir / "generated_text.csv")
    keep_indices = set(
        gen.loc[gen["generated_token_count"] >= min_generated_tokens, "sample_index"].astype(int).tolist()
    )
    mean_df = pd.read_csv(analysis_dir / "rope_step_mean.csv").copy()
    mean_df["n_samples"] = len(keep_indices)
    stats = {
        "total_samples": int(len(gen)),
        "kept_samples": int(len(keep_indices)),
        "min_generated_tokens": int(min_generated_tokens),
    }
    bin_means: dict[str, float] = {}
    for name, lo, hi in BIN_SPECS:
        sub = mean_df[(mean_df["step"] >= lo) & (mean_df["step"] <= hi)]
        if len(sub) > 0:
            bin_means[name] = float(sub["image2_over_image1"].mean())
    return mean_df, stats, bin_means


def plot_family(
    *,
    output_dir: Path,
    dataset_label: str,
    suffix: str,
    min_generated_tokens: int,
    blank_analysis_dir: Path,
    blank_rope_dir: Path,
    swap_analysis_dir: Path,
    swap_rope_dir: Path,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True, sharey=False)
    fig.suptitle(
        f"{dataset_label}: rope-align only faithful image2-image1 gap ({suffix}, min decode len {min_generated_tokens})",
        fontsize=14,
    )
    records = []
    configs = [
        ("blank", blank_analysis_dir, blank_rope_dir),
        ("dataset_swap", swap_analysis_dir, swap_rope_dir),
    ]
    for ax, (corruption, analysis_dir, raw_rope_dir) in zip(axes, configs):
        mean_df, stats, bin_means = load_rope_mean(analysis_dir, raw_rope_dir, min_generated_tokens)
        for idx, (_, lo, hi) in enumerate(BIN_SPECS):
            ax.axvspan(lo, hi, color=("#f3f4f6" if idx % 2 == 0 else "#e5e7eb"), alpha=0.45, linewidth=0)
        ax.plot(mean_df["step"], mean_df["image2_over_image1"], color="#0f766e", linewidth=2.2)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{corruption} (kept {stats['kept_samples']}/{stats['total_samples']})")
        ax.set_xlabel("decode step")
        ax.set_ylabel("image2 - image1")
        ax.grid(alpha=0.18, linewidth=0.6)
        note = " | ".join(f"{name}:{val:.3f}" for name, val in bin_means.items())
        ax.text(
            0.02,
            0.98,
            note,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor="none"),
        )
        for step, n_samples in zip(mean_df["step"], mean_df["n_samples"]):
            records.append(
                {
                    "dataset": dataset_label,
                    "corruption": corruption,
                    "suffix": suffix,
                    "step": int(step),
                    "image2_over_image1": float(
                        mean_df.loc[mean_df["step"] == step, "image2_over_image1"].iloc[0]
                    ),
                    "n_samples": int(n_samples),
                    **stats,
                }
            )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_png = output_dir / f"{dataset_label.lower()}_rope_only_gap_shapes_{suffix}_min{min_generated_tokens}.png"
    out_csv = output_dir / f"{dataset_label.lower()}_rope_only_gap_shapes_{suffix}_min{min_generated_tokens}.csv"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    pd.DataFrame(records).to_csv(out_csv, index=False)
    return out_png, out_csv


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for suffix, blank_dir, swap_dir in [
        ("logprob", Path(args.blank_analysis_logprob_dir), Path(args.swap_analysis_logprob_dir)),
        ("margin", Path(args.blank_analysis_margin_dir), Path(args.swap_analysis_margin_dir)),
    ]:
        png, csv = plot_family(
            output_dir=output_dir,
            dataset_label=args.dataset_label,
            suffix=suffix,
            min_generated_tokens=args.min_generated_tokens,
            blank_analysis_dir=blank_dir,
            blank_rope_dir=Path(args.blank_rope_dir),
            swap_analysis_dir=swap_dir,
            swap_rope_dir=Path(args.swap_rope_dir),
        )
        manifest.append({"suffix": suffix, "png": str(png), "csv": str(csv)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
