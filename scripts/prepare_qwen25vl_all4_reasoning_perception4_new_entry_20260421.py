#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


MODELS = ["qwen25vl_72b", "qwen25vl_32b", "qwen25vl_7b", "qwen25vl_3b"]
DATASETS = ["MMMU_DEV_VAL_SINGLE_IMAGE", "WeMath", "MMBench_DEV_EN_V11", "MMStar"]
MODES = [
    "image_text",
    "text_image",
    "image_image_text",
    "image_text_image",
    "image_text_text",
    "image_text_image_text",
]
OUT_DIR = Path("scripts/configs/task_manifests/qwen25vl_all4_reasoning_perception4_new_entry_20260421")


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for model_idx, model in enumerate(MODELS):
        for mode_idx, mode in enumerate(MODES):
            for dataset_idx, dataset in enumerate(DATASETS):
                node = (mode_idx + dataset_idx + model_idx) % 4
                rows.append(
                    {
                        "model": model,
                        "policy": "default",
                        "mode": mode,
                        "transform": "baseline",
                        "dataset": dataset,
                        "node": str(node),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "policy", "mode", "transform", "dataset", "node"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_rows()
    write_csv(OUT_DIR / "all_tasks.csv", rows)
    for node in range(4):
        write_csv(OUT_DIR / f"node{node}_tasks.csv", [row for row in rows if row["node"] == str(node)])


if __name__ == "__main__":
    main()
