#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


DATASETS = [
    ("MathVision", 3040),
    ("DynaMath", 5010),
    ("LogicVista", 447),
    ("VisualPuzzles", 1168),
    ("AI2D_TEST", 3088),
    ("OCRBench", 1000),
    ("SEEDBench2_Plus", 2277),
]

TASK1_TRANSFORMS = [
    "baseline",
    "shift_right_halfpatch_wrap",
    "shift_down_halfpatch_wrap",
    "shift_left_halfpatch_wrap",
    "shift_up_halfpatch_wrap",
]

TASK2_TRANSFORMS = [
    "shift_right_real_half_patch_wrap",
    "shift_down_real_half_patch_wrap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect task1/task2 result status from the 20260404 result root.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--task1-out", type=Path, required=True)
    parser.add_argument("--task2-out", type=Path, required=True)
    parser.add_argument("--status-out", type=Path, required=True)
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_parent(path)
    fieldnames = [
        "model_key",
        "registry_name",
        "policy",
        "mode",
        "transform",
        "dataset",
        "expected_count",
        "infer_complete",
        "eval_complete",
        "metric_value",
        "metric_kind",
        "infer_file",
        "metric_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_metric_file(model_dir: Path, dataset: str) -> Path | None:
    prefix = f"Qwen2VLChatReplay_{dataset}"
    candidates: list[Path] = []
    for path in model_dir.iterdir():
        if not path.is_file():
            continue
        if not path.name.startswith(prefix):
            continue
        if "answer_format" in path.name:
            continue
        if path.suffix not in {".csv", ".json"}:
            continue
        if ("score" not in path.name) and ("acc" not in path.name):
            continue
        candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return candidates[0]


def maybe_scale(value: float) -> float:
    return value * 100.0 if value <= 1.0 else value


def read_metric_value(path: Path) -> float | None:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ["Final Score Norm", "score", "Score", "Overall"]:
                if key in data:
                    return float(data[key])
        return None

    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None

    header = rows[0].keys()
    if "Setting" in header and "Overall" in header:
        for row in rows:
            if str(row.get("Setting", "")).strip() == "Average":
                return maybe_scale(float(row["Overall"]))
        return maybe_scale(float(rows[0]["Overall"]))

    if "Task&Skill" in header and "acc" in header:
        for row in rows:
            if str(row.get("Task&Skill", "")).strip() == "Overall":
                return maybe_scale(float(row["acc"]))
        return maybe_scale(float(rows[0]["acc"]))

    if "Subject" in header and "acc" in header:
        for row in rows:
            if str(row.get("Subject", "")).strip() == "Overall":
                return maybe_scale(float(row["acc"]))
        return maybe_scale(float(rows[0]["acc"]))

    if "group" in header and "acc" in header:
        for row in rows:
            if str(row.get("group", "")).strip() == "overall":
                return maybe_scale(float(row["acc"]))
        return maybe_scale(float(rows[0]["acc"]))

    if "split" in header and "Overall" in header:
        return maybe_scale(float(rows[0]["Overall"]))

    if "Overall" in header:
        return maybe_scale(float(rows[0]["Overall"]))

    if "acc" in header:
        return maybe_scale(float(rows[0]["acc"]))

    return None


def collect_rows(results_root: Path, transforms: Iterable[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for transform in transforms:
        task_root = results_root / "default" / "image_text_image" / transform / "qwen25vl_32b"
        model_dir = task_root / "Qwen2VLChatReplay"
        for dataset, expected_count in DATASETS:
            infer_file = model_dir / f"Qwen2VLChatReplay_{dataset}.xlsx"
            metric_file = choose_metric_file(model_dir, dataset) if model_dir.exists() else None
            metric_value = read_metric_value(metric_file) if metric_file is not None else None
            rows.append(
                {
                    "model_key": "qwen25vl_32b",
                    "registry_name": "Qwen2VLChatReplay",
                    "policy": "default",
                    "mode": "image_text_image",
                    "transform": transform,
                    "dataset": dataset,
                    "expected_count": expected_count,
                    "infer_complete": infer_file.exists(),
                    "eval_complete": metric_value is not None,
                    "metric_value": "" if metric_value is None else f"{metric_value:.6f}",
                    "metric_kind": "" if metric_file is None else metric_file.suffix.lstrip("."),
                    "infer_file": str(infer_file) if infer_file.exists() else "",
                    "metric_file": str(metric_file) if metric_file is not None else "",
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    task1_rows = collect_rows(args.results_root, TASK1_TRANSFORMS)
    task2_rows = collect_rows(args.results_root, TASK2_TRANSFORMS)
    write_csv(args.task1_out, task1_rows)
    write_csv(args.task2_out, task2_rows)
    write_csv(args.status_out, task1_rows + task2_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
