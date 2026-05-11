#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


BASELINE_MODE_ROWS = [
    ("IQ", "image_text"),
    ("QI", "text_image"),
    ("IIQ", "image_image_text"),
    ("IQQ", "image_text_text"),
    ("IQIQ", "image_text_image_text"),
]

IQI_TRANSFORM_ROWS = [
    ("IQI", "baseline", "-"),
    ("", "mask10_white", "mask10_white"),
    ("", "mask20_white", "mask20_white"),
    ("", "blank", "blank"),
    ("", "rotate180", "rotate180"),
    ("", "shift_down_halfpatch_wrap", "shift_down_halfpatch_wrap"),
    ("", "shift_down_onepatch_wrap", "shift_down_onepatch_wrap"),
    ("", "shift_right_halfpatch_wrap", "shift_right_halfpatch_wrap"),
    ("", "shift_right_onepatch_wrap", "shift_right_onepatch_wrap"),
    ("", "zoom_1p5_uncropped", "zoom_1p5_uncropped"),
]

DATASET_COLUMNS = [
    ("MathVision", "MathVision"),
    ("DynaMath", "DynaMath"),
    ("LogicVista", "LogicVista"),
    ("VisualPuzzles", "VisualPuzzles"),
    ("MathVista", "MathVista_MINI"),
    ("VisuLogic", "VisuLogic"),
    ("AI2D_TEST", "AI2D_TEST"),
    ("OCRBench", "OCRBench"),
    ("SEEDBench2_Plus", "SEEDBench2_Plus"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Qwen2.5-VL summary table in the paper-style layout.")
    parser.add_argument("--results-long", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    return parser.parse_args()


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def load_scores(path: Path) -> dict[tuple[str, str, str], float]:
    scores: dict[tuple[str, str, str], float] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("policy") != "default":
                continue
            mode = row.get("mode", "")
            transform = row.get("transform", "")
            dataset = row.get("dataset", "")
            metric_raw = row.get("metric_value", "")
            if not metric_raw.strip():
                continue
            scores[(mode, transform, dataset)] = float(metric_raw)
    return scores


def build_rows(scores: dict[tuple[str, str, str], float]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for setting, mode in BASELINE_MODE_ROWS:
        out = {"setting": setting, "transform": "-"}
        for col_name, dataset_key in DATASET_COLUMNS:
            out[col_name] = fmt(scores.get((mode, "baseline", dataset_key)))
        rows.append(out)

    for setting, transform_key, transform_label in IQI_TRANSFORM_ROWS:
        out = {"setting": setting, "transform": transform_label}
        for col_name, dataset_key in DATASET_COLUMNS:
            out[col_name] = fmt(scores.get(("image_text_image", transform_key, dataset_key)))
        rows.append(out)
    return rows


def write_csv_table(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["setting", "transform"] + [col for col, _ in DATASET_COLUMNS]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["setting", "transform"] + [col for col, _ in DATASET_COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(h, "") for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    scores = load_scores(args.results_long)
    rows = build_rows(scores)
    write_csv_table(args.out_csv, rows)
    if args.out_md is not None:
        write_markdown_table(args.out_md, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
