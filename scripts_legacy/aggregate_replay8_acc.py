#!/usr/bin/env python3
"""Aggregate replay8 *_acc.csv files into a single summary CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

SUBSETS = [
    "ReplayIconA_L2R",
    "ReplayIconA_R2L",
    "ReplayIconB_L2R",
    "ReplayIconB_R2L",
    "ReplayShapeA_L2R",
    "ReplayShapeA_R2L",
    "ReplayShapeB_L2R",
    "ReplayShapeB_R2L",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate replay8 acc results into one CSV.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="./runs/standard/20260301/qwen2_qwen25_minicpm_replay8_2node16gpu",
        help="Root run directory containing task subdirs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output CSV path. Default: <run_dir>/replay8_summary.csv",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./exp_debug/replay_8subsets_v1",
        help="Dataset root to read per-subset num_objects metadata.",
    )
    return parser.parse_args()


def read_overall_acc(csv_path: Path) -> float | None:
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # First data row usually has Overall; split col may be "none" or similar
            if "Overall" in row:
                try:
                    return float(row["Overall"])
                except (ValueError, TypeError):
                    return None
    return None


def read_num_objects_mean(tsv_path: Path) -> float | None:
    if not tsv_path.exists():
        return None
    vals: list[float] = []
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                vals.append(float(row.get("num_objects", "")))
            except (TypeError, ValueError):
                continue
    if not vals:
        return None
    return sum(vals) / len(vals)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_dir / "replay8_summary.csv"
    )

    subset_numobj_means: dict[str, str] = {}
    numobj_vals: list[float] = []
    for ds in SUBSETS:
        m = read_num_objects_mean(data_root / f"{ds}.tsv")
        if m is None:
            subset_numobj_means[ds] = ""
        else:
            subset_numobj_means[ds] = f"{m:.2f}"
            numobj_vals.append(m)
    numobj_overall = f"{(sum(numobj_vals) / len(numobj_vals)):.2f}" if numobj_vals else ""

    # Task dirs: {model}__{replay_mode}__last1
    results: list[dict] = []
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
        acc_dir = task_dir / "output" / "Qwen2VLChatReplay"
        if not acc_dir.exists():
            acc_dir = task_dir / "output"
            # find first dir that has *_acc.csv
            for sub in acc_dir.iterdir():
                if sub.is_dir():
                    acc_dir = sub
                    break

        row = {"model": model, "replay_mode": replay_mode}
        accs = []
        for ds in SUBSETS:
            candidates = list(acc_dir.glob(f"*_{ds}_acc.csv"))
            if not candidates:
                row[ds] = ""
                continue
            acc = read_overall_acc(candidates[0])
            if acc is not None:
                row[ds] = f"{acc:.4f}"
                accs.append(acc)
            else:
                row[ds] = ""
        if accs:
            row["mean"] = f"{sum(accs) / len(accs):.4f}"
        else:
            row["mean"] = ""
        for ds in SUBSETS:
            row[f"{ds}_num_objects_mean"] = subset_numobj_means.get(ds, "")
        row["num_objects_mean"] = numobj_overall
        results.append(row)

    cols = (
        ["model", "replay_mode"]
        + SUBSETS
        + ["mean"]
        + [f"{ds}_num_objects_mean" for ds in SUBSETS]
        + ["num_objects_mean"]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"[DONE] {out_path} ({len(results)} rows)")


if __name__ == "__main__":
    main()
