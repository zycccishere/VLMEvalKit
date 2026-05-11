#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


FILE_RE = re.compile(r"^Qwen2VLChatReplay_(.+?)_gpt-4o_result\.xlsx$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate replay8 accuracy trend by num_objects.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="./runs/standard/20260301/qwen2_qwen25_minicpm_replay8_2node16gpu",
        help="Run directory containing task subdirectories.",
    )
    parser.add_argument(
        "--output-long",
        type=str,
        default="",
        help="Output long-format CSV. Default: <run_dir>/replay8_num_objects_trend_long.csv",
    )
    parser.add_argument(
        "--output-wide",
        type=str,
        default="",
        help="Output wide-format CSV. Default: <run_dir>/replay8_num_objects_trend_wide.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_long = (
        Path(args.output_long).expanduser().resolve()
        if args.output_long
        else run_dir / "replay8_num_objects_trend_long.csv"
    )
    out_wide = (
        Path(args.output_wide).expanduser().resolve()
        if args.output_wide
        else run_dir / "replay8_num_objects_trend_wide.csv"
    )

    rows: list[dict] = []
    for task_dir in sorted(run_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        name = task_dir.name
        if "__" not in name or not name.endswith("__last1"):
            continue
        parts = name.rsplit("__", 2)
        if len(parts) != 3:
            continue
        model, replay_mode = parts[0], parts[1]

        result_dir = task_dir / "output" / "Qwen2VLChatReplay"
        if not result_dir.exists():
            continue

        for p in sorted(result_dir.glob("*_gpt-4o_result.xlsx")):
            m = FILE_RE.match(p.name)
            if m is None:
                continue
            dataset = m.group(1)
            try:
                df = pd.read_excel(p)
            except Exception:
                continue
            if "num_objects" not in df.columns or "hit" not in df.columns:
                continue

            df = df[["num_objects", "hit"]].copy()
            df = df.dropna(subset=["num_objects", "hit"])
            if df.empty:
                continue
            df["num_objects"] = df["num_objects"].astype(int)
            df["hit"] = df["hit"].astype(float)

            grouped = df.groupby("num_objects")["hit"].agg(["mean", "count"]).reset_index()
            for _, r in grouped.iterrows():
                rows.append(
                    {
                        "model": model,
                        "replay_mode": replay_mode,
                        "dataset": dataset,
                        "num_objects": int(r["num_objects"]),
                        "acc": float(r["mean"]),
                        "count": int(r["count"]),
                    }
                )

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        out_long.parent.mkdir(parents=True, exist_ok=True)
        out_long.write_text("model,replay_mode,dataset,num_objects,acc,count\n", encoding="utf-8")
        out_wide.write_text("model,replay_mode,dataset\n", encoding="utf-8")
        print(f"[DONE] {out_long} (0 rows)")
        print(f"[DONE] {out_wide} (0 rows)")
        return

    long_df = long_df.sort_values(["model", "replay_mode", "dataset", "num_objects"]).reset_index(drop=True)
    out_long.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_long, index=False)

    wide = long_df.pivot_table(
        index=["model", "replay_mode", "dataset"],
        columns="num_objects",
        values="acc",
        aggfunc="first",
    )
    wide = wide.reset_index()
    wide.columns = [
        c if isinstance(c, str) else f"num_objects_{int(c)}_acc"
        for c in wide.columns
    ]
    wide.to_csv(out_wide, index=False)

    print(f"[DONE] {out_long} ({len(long_df)} rows)")
    print(f"[DONE] {out_wide} ({len(wide)} rows)")


if __name__ == "__main__":
    main()
