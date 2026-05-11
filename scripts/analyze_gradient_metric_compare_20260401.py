#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare standard vs rope gradient-style metrics.")
    parser.add_argument("--standard-dir", required=True)
    parser.add_argument("--rope-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def load_metric_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path / "metric_rows.csv")
    return df


def main() -> int:
    args = build_parser().parse_args()
    standard_dir = Path(args.standard_dir)
    rope_dir = Path(args.rope_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    std = load_metric_rows(standard_dir)
    rope = load_metric_rows(rope_dir)
    merged = std.merge(
        rope,
        on=["sample_index", "score_kind", "metric"],
        suffixes=("_standard", "_rope"),
    )
    merged["gap_delta"] = merged["image2_over_image1_rope"] - merged["image2_over_image1_standard"]
    merged.to_csv(output_dir / "merged_sample_metrics.csv", index=False)

    summary_rows = []
    for (score_kind, metric), group in merged.groupby(["score_kind", "metric"], dropna=False):
        summary_rows.append(
            {
                "score_kind": score_kind,
                "metric": metric,
                "n_samples": int(group["sample_index"].nunique()),
                "standard_gap_mean": float(group["image2_over_image1_standard"].mean()),
                "rope_gap_mean": float(group["image2_over_image1_rope"].mean()),
                "delta_gap_mean": float(group["gap_delta"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["score_kind", "metric"])
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    if not summary_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
        for ax, score_kind in zip(axes, sorted(summary_df["score_kind"].unique())):
            sub = summary_df[summary_df["score_kind"] == score_kind].copy()
            x = range(len(sub))
            ax.bar([i - 0.2 for i in x], sub["standard_gap_mean"], width=0.4, label="standard")
            ax.bar([i + 0.2 for i in x], sub["rope_gap_mean"], width=0.4, label="rope_align")
            ax.set_xticks(list(x))
            ax.set_xticklabels(sub["metric"], rotation=20, ha="right")
            ax.set_title(score_kind)
            ax.set_ylabel("image2 - image1")
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
            ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "compare_gradient_metrics.png", dpi=220)
        plt.close(fig)

    (output_dir / "summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
