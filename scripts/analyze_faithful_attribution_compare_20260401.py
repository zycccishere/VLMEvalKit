#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare standard vs rope-align faithful attribution outputs."
    )
    parser.add_argument("--standard-dir", required=True)
    parser.add_argument("--rope-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-suffix", default="logprob", choices=["logprob", "margin"])
    return parser


def load_shapley(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "step_shapley.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_step_scores(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "step_scores.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def compute_two_player_shapley(
    *,
    full_score: float,
    no_image1_score: float,
    no_image2_score: float,
    no_both_score: float,
) -> tuple[float, float, float]:
    phi_image1 = 0.5 * ((full_score - no_image2_score) + (no_image1_score - no_both_score))
    phi_image2 = 0.5 * ((full_score - no_image1_score) + (no_image2_score - no_both_score))
    interaction = full_score - no_image1_score - no_image2_score + no_both_score
    return phi_image1, phi_image2, interaction


def ensure_metric_shapley(run_dir: Path, suffix: str) -> pd.DataFrame:
    shapley_df = load_shapley(run_dir)
    needed = [
        f"phi_image1_{suffix}",
        f"phi_image2_{suffix}",
        f"interaction_{suffix}",
    ]
    if all(col in shapley_df.columns for col in needed):
        return shapley_df

    score_df = load_step_scores(run_dir)
    value_col = "target_logprob" if suffix == "logprob" else "logit_margin"
    rows = []
    group_cols = ["sample_index", "step", "token_id", "token_text"]
    for keys, group in score_df.groupby(group_cols, dropna=False):
        score_map = dict(zip(group["condition"], group[value_col]))
        missing = [name for name in ("full", "no_image1", "no_image2", "no_both") if name not in score_map]
        if missing:
            raise KeyError(f"missing conditions {missing} in {run_dir}")
        phi1, phi2, interaction = compute_two_player_shapley(
            full_score=float(score_map["full"]),
            no_image1_score=float(score_map["no_image1"]),
            no_image2_score=float(score_map["no_image2"]),
            no_both_score=float(score_map["no_both"]),
        )
        row = {
            "sample_index": int(keys[0]),
            "step": int(keys[1]),
            "token_id": int(keys[2]),
            "token_text": keys[3],
            f"phi_image1_{suffix}": phi1,
            f"phi_image2_{suffix}": phi2,
            f"interaction_{suffix}": interaction,
            f"full_{suffix}": float(score_map["full"]),
            f"no_image1_{suffix}": float(score_map["no_image1"]),
            f"no_image2_{suffix}": float(score_map["no_image2"]),
            f"no_both_{suffix}": float(score_map["no_both"]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_steps(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    cols = [
        f"phi_image1_{suffix}",
        f"phi_image2_{suffix}",
        f"interaction_{suffix}",
    ]
    keep = ["step", *cols]
    out = df[keep].groupby("step", as_index=False).mean(numeric_only=True)
    out["image2_over_image1"] = out[f"phi_image2_{suffix}"] - out[f"phi_image1_{suffix}"]
    return out


def plot_step_curves(standard_df: pd.DataFrame, rope_df: pd.DataFrame, suffix: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in [
        (axes[0], f"phi_image1_{suffix}", "Image1 Contribution"),
        (axes[1], f"phi_image2_{suffix}", "Image2 Contribution"),
        (axes[2], f"interaction_{suffix}", "Interaction"),
    ]:
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
    for ax, metric, title in [
        (axes[0], f"phi_image1_{suffix}", "Rope - Standard: Image1"),
        (axes[1], f"phi_image2_{suffix}", "Rope - Standard: Image2"),
        (axes[2], f"interaction_{suffix}", "Rope - Standard: Interaction"),
    ]:
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

    standard_df = summarize_steps(ensure_metric_shapley(standard_dir, args.score_suffix), args.score_suffix)
    rope_df = summarize_steps(ensure_metric_shapley(rope_dir, args.score_suffix), args.score_suffix)
    merged = standard_df.merge(rope_df, on="step", suffixes=("_standard", "_rope"))

    delta_rows = []
    for _, row in merged.iterrows():
        delta_row = {"step": int(row["step"])}
        for key in [
            f"phi_image1_{args.score_suffix}",
            f"phi_image2_{args.score_suffix}",
            f"interaction_{args.score_suffix}",
            "image2_over_image1",
        ]:
            delta_row[key] = float(row[f"{key}_rope"] - row[f"{key}_standard"])
        delta_rows.append(delta_row)
    delta_df = pd.DataFrame(delta_rows)

    standard_df.to_csv(output_dir / "standard_step_mean.csv", index=False)
    rope_df.to_csv(output_dir / "rope_step_mean.csv", index=False)
    delta_df.to_csv(output_dir / "rope_minus_standard_step_mean.csv", index=False)

    plot_step_curves(
        standard_df,
        rope_df,
        args.score_suffix,
        output_dir / f"compare_faithful_contribution_{args.score_suffix}.png",
    )
    plot_delta(
        delta_df,
        args.score_suffix,
        output_dir / f"delta_faithful_contribution_{args.score_suffix}.png",
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
