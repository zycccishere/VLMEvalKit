#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare attention-side image2/image1 trends against faithful output-level contribution trends."
    )
    parser.add_argument("--attention-dir", required=True)
    parser.add_argument("--faithful-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--layer",
        default="last",
        help="Attention layer to compare. Use an integer layer id or 'last'.",
    )
    return parser


def resolve_attention_layer(df: pd.DataFrame, layer_arg: str) -> int:
    layers = sorted(df["layer"].dropna().astype(int).unique().tolist())
    if not layers:
        raise ValueError("No attention layers found.")
    if layer_arg == "last":
        return layers[-1]
    return int(layer_arg)


def load_attention(attention_dir: Path, layer_arg: str) -> tuple[int, pd.DataFrame]:
    path = attention_dir / "rope_minus_standard_decode_layer_step_summary.csv"
    df = pd.read_csv(path)
    layer = resolve_attention_layer(df, layer_arg)
    keep = df[df["layer"] == layer][["step", "image2_over_image1_ratio_delta"]].copy()
    keep = keep.rename(columns={"image2_over_image1_ratio_delta": "attention_delta"})
    return layer, keep


def load_faithful(faithful_dir: Path) -> pd.DataFrame:
    path = faithful_dir / "rope_minus_standard_step_mean.csv"
    df = pd.read_csv(path)
    return df[["step", "image2_over_image1"]].rename(columns={"image2_over_image1": "faithful_delta"})


def plot_curves(df: pd.DataFrame, layer: int, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(df["step"], df["attention_delta"], color="#4c72b0", linewidth=1.8)
    axes[0].axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel(f"attn delta L{layer}")
    axes[0].set_title("Attention routing delta vs faithful contribution delta")

    axes[1].plot(df["step"], df["faithful_delta"], color="#dd8452", linewidth=1.8)
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("faithful delta")
    axes[1].set_xlabel("decode step")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, layer: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["attention_delta"], df["faithful_delta"], alpha=0.8, s=18)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"attention delta L{layer}")
    ax.set_ylabel("faithful delta")
    ax.set_title("Per-step delta agreement")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer, attention_df = load_attention(Path(args.attention_dir), args.layer)
    faithful_df = load_faithful(Path(args.faithful_dir))
    merged = attention_df.merge(faithful_df, on="step", how="inner").sort_values("step")
    merged["attention_sign"] = merged["attention_delta"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    merged["faithful_sign"] = merged["faithful_delta"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    merged["sign_agree"] = (merged["attention_sign"] == merged["faithful_sign"]).astype(int)

    corr = float(merged["attention_delta"].corr(merged["faithful_delta"])) if len(merged) > 1 else float("nan")
    summary = {
        "attention_dir": str(Path(args.attention_dir)),
        "faithful_dir": str(Path(args.faithful_dir)),
        "layer": layer,
        "step_count": int(len(merged)),
        "mean_attention_delta": float(merged["attention_delta"].mean()),
        "mean_faithful_delta": float(merged["faithful_delta"].mean()),
        "delta_correlation": corr,
        "sign_agreement_rate": float(merged["sign_agree"].mean()) if len(merged) else float("nan"),
    }

    merged.to_csv(output_dir / "attention_vs_faithful_step_compare.csv", index=False)
    plot_curves(merged, layer, output_dir / "attention_vs_faithful_curves.png")
    plot_scatter(merged, layer, output_dir / "attention_vs_faithful_scatter.png")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
