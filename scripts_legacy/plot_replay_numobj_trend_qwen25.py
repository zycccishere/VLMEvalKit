#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot num_objects trend for Qwen2.5-VL-7B-Instruct (one figure per dataset)."
    )
    parser.add_argument(
        "--trend-csv",
        type=str,
        default="./runs/standard/20260301/qwen2_qwen25_minicpm_replay8_2node16gpu/replay8_num_objects_trend_long.csv",
        help="Input long-format trend CSV from aggregate_replay8_num_objects_trend.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./runs/standard/20260301/qwen2_qwen25_minicpm_replay8_2node16gpu/plots_qwen25_numobj",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen2.5-VL-7B-Instruct",
        help="Model name to filter.",
    )
    return parser.parse_args()


def sanitize_filename(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def main() -> None:
    args = parse_args()
    trend_csv = Path(args.trend_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(trend_csv)
    needed_cols = {"model", "replay_mode", "dataset", "num_objects", "acc"}
    if not needed_cols.issubset(set(df.columns)):
        raise ValueError(f"Missing required columns in {trend_csv}. Need: {sorted(needed_cols)}")

    sub = df[df["model"] == args.model].copy()
    if sub.empty:
        raise ValueError(f"No rows for model={args.model} in {trend_csv}")

    sub["num_objects"] = sub["num_objects"].astype(int)
    sub["acc"] = sub["acc"].astype(float)

    datasets = sorted(sub["dataset"].unique().tolist())
    mode_order = ["none", "image_text_text", "image_text_image", "image_text_image_text", "image_image_text"]

    for ds in datasets:
        ds_df = sub[sub["dataset"] == ds].copy()
        if ds_df.empty:
            continue

        plt.figure(figsize=(8.2, 5.0))
        for mode in mode_order:
            md = ds_df[ds_df["replay_mode"] == mode].sort_values("num_objects")
            if md.empty:
                continue
            plt.plot(
                md["num_objects"],
                md["acc"],
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=mode,
            )

        plt.title(f"{args.model} | {ds}")
        plt.xlabel("num_objects")
        plt.ylabel("accuracy")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best", fontsize=9)
        plt.tight_layout()

        out_png = output_dir / f"{sanitize_filename(ds)}.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[DONE] {out_png}")


if __name__ == "__main__":
    main()
