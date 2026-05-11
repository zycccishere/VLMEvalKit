#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare standard vs rope-align grad×input outputs."
    )
    parser.add_argument("--standard-dir", required=True)
    parser.add_argument("--rope-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-suffix", default="logprob", choices=["logprob", "margin"])
    return parser


def load_attr(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "step_attribution.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def summarize_steps(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    cols = [
        f"phi_image1_{suffix}",
        f"phi_image2_{suffix}",
        f"text_{suffix}",
    ]
    present = ["step", *[col for col in cols if col in df.columns]]
    out = df[present].groupby("step", as_index=False).mean(numeric_only=True)
    out["image2_over_image1"] = out[f"phi_image2_{suffix}"] - out[f"phi_image1_{suffix}"]
    return out


def plot_curves(standard_df: pd.DataFrame, rope_df: pd.DataFrame, suffix: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [
        (f"phi_image1_{suffix}", "Image1 Grad×Input"),
        (f"phi_image2_{suffix}", "Image2 Grad×Input"),
        ("image2_over_image1", "Gap: Image2 - Image1"),
    ]
    for ax, metric, title in metrics:
        ax.plot(standard_df["step"], standard_df[metric], label="standard")
        ax.plot(rope_df["step"], rope_df[metric], label="rope_align")
        ax.set_title(title)
        ax.set_xlabel("decode step")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_delta(delta_df: pd.DataFrame, suffix: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [
        (f"phi_image1_{suffix}", "Rope - Standard: Image1"),
        (f"phi_image2_{suffix}", "Rope - Standard: Image2"),
        ("image2_over_image1", "Rope - Standard: Gap"),
    ]
    for ax, metric, title in metrics:
        ax.plot(delta_df["step"], delta_df[metric])
        ax.set_title(title)
        ax.set_xlabel("decode step")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    standard_dir = Path(args.standard_dir)
    rope_dir = Path(args.rope_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    standard_df = summarize_steps(load_attr(standard_dir), args.score_suffix)
    rope_df = summarize_steps(load_attr(rope_dir), args.score_suffix)
    merged = standard_df.merge(rope_df, on="step", suffixes=("_standard", "_rope"))

    delta_rows = []
    for _, row in merged.iterrows():
        delta_row = {"step": int(row["step"])}
        for key in [
            f"phi_image1_{args.score_suffix}",
            f"phi_image2_{args.score_suffix}",
            "image2_over_image1",
        ]:
            delta_row[key] = float(row[f"{key}_rope"] - row[f"{key}_standard"])
        delta_rows.append(delta_row)
    delta_df = pd.DataFrame(delta_rows)

    standard_df.to_csv(output_dir / "standard_step_mean.csv", index=False)
    rope_df.to_csv(output_dir / "rope_step_mean.csv", index=False)
    delta_df.to_csv(output_dir / "rope_minus_standard_step_mean.csv", index=False)

    plot_curves(
        standard_df,
        rope_df,
        args.score_suffix,
        output_dir / f"compare_gradxinput_{args.score_suffix}.png",
    )
    plot_delta(
        delta_df,
        args.score_suffix,
        output_dir / f"delta_gradxinput_{args.score_suffix}.png",
    )

    summary = {
        "score_suffix": args.score_suffix,
        "standard_dir": str(standard_dir),
        "rope_dir": str(rope_dir),
        "output_dir": str(output_dir),
        "standard_mean": standard_df.mean(numeric_only=True).to_dict(),
        "rope_mean": rope_df.mean(numeric_only=True).to_dict(),
        "delta_mean": delta_df.mean(numeric_only=True).to_dict(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
