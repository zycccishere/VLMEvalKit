#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


REASONING_DATASETS = [
    "DynaMath",
    "LogicVista",
    "MathVision",
    "VisualPuzzles",
]

NON_REASONING_DATASETS = [
    "AI2D_TEST",
    "SEEDBench2_Plus",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine baseline-wrong -> shift-correct cases from the 20260404 Qwen32B image2 result root."
    )
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=["shift_right_halfpatch_wrap", "shift_down_halfpatch_wrap"],
    )
    parser.add_argument("--per-transform-total", type=int, default=20)
    parser.add_argument("--reasoning-quota", type=int, default=10)
    parser.add_argument("--non-reasoning-quota", type=int, default=10)
    return parser


def resolve_eval_file(task_root: Path, dataset: str) -> Path:
    candidates = [
        task_root / f"Qwen2VLChatReplay_{dataset}_gpt-4o-mini.xlsx",
        task_root / f"Qwen2VLChatReplay_{dataset}_gpt4o-mini.xlsx",
        task_root / f"Qwen2VLChatReplay_{dataset}_gpt-4o-mini_result.xlsx",
        task_root / f"Qwen2VLChatReplay_{dataset}_gpt4o-mini_result.xlsx",
        task_root / f"Qwen2VLChatReplay_{dataset}.xlsx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No eval/infer file found for dataset={dataset} under {task_root}")


def load_eval_df(task_root: Path, dataset: str) -> pd.DataFrame:
    path = resolve_eval_file(task_root, dataset)
    return pd.read_excel(path)


def normalize_correct(dataset: str, df: pd.DataFrame) -> pd.Series:
    if dataset == "DynaMath":
        return df["correct"].fillna(False).astype(bool)
    if dataset == "LogicVista":
        return df["hit"].fillna(0).astype(int).astype(bool)
    if dataset == "MathVision":
        return df["hit_score"].fillna(0).astype(float) > 0.5
    if dataset in {"AI2D_TEST", "SEEDBench2_Plus"}:
        return df["hit"].fillna(0).astype(int).astype(bool)
    if dataset == "VisualPuzzles":
        pred = df["extracted_answer"].fillna("").astype(str).str.strip().str.upper()
        gold = df["answer"].fillna("").astype(str).str.strip().str.upper()
        return pred == gold
    raise KeyError(f"Unsupported dataset for correctness normalization: {dataset}")


def dataset_group(dataset: str) -> str:
    if dataset in REASONING_DATASETS:
        return "reasoning"
    if dataset in NON_REASONING_DATASETS:
        return "non_reasoning"
    raise KeyError(f"Unsupported dataset group for {dataset}")


def merged_improvements(
    *,
    result_root: Path,
    transform: str,
    dataset: str,
) -> list[dict[str, Any]]:
    baseline_root = result_root / "default" / "image_text_image" / "baseline" / "qwen25vl_32b" / "Qwen2VLChatReplay"
    shift_root = result_root / "default" / "image_text_image" / transform / "qwen25vl_32b" / "Qwen2VLChatReplay"

    baseline = load_eval_df(baseline_root, dataset).copy()
    shift = load_eval_df(shift_root, dataset).copy()
    baseline["_correct"] = normalize_correct(dataset, baseline)
    shift["_correct"] = normalize_correct(dataset, shift)

    if "index" not in baseline.columns or "index" not in shift.columns:
        raise KeyError(f"Dataset {dataset} does not expose a stable `index` column.")

    keep_cols = ["index", "_correct"]
    for col in ("question", "answer"):
        if col in baseline.columns:
            keep_cols.append(col)
    merged = baseline[keep_cols].merge(
        shift[["index", "_correct"]],
        on="index",
        suffixes=("_baseline", "_shift"),
    )
    improved = merged[(~merged["_correct_baseline"]) & (merged["_correct_shift"])].copy()
    improved = improved.sort_values("index").reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for _, row in improved.iterrows():
        rows.append(
            {
                "id": f"{transform}__{dataset}__idx{int(row['index'])}",
                "shift_transform": transform,
                "source_dataset": dataset,
                "group": dataset_group(dataset),
                "source_index": int(row["index"]),
                "question": str(row.get("question", "")),
                "answer": str(row.get("answer", "")),
            }
        )
    return rows


def round_robin_pick(
    *,
    candidates_by_dataset: dict[str, list[dict[str, Any]]],
    quota: int,
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    cursors = {dataset: 0 for dataset in candidates_by_dataset}
    active = [dataset for dataset, rows in candidates_by_dataset.items() if rows]
    while len(chosen) < quota and active:
        next_active: list[str] = []
        for dataset in active:
            rows = candidates_by_dataset[dataset]
            idx = cursors[dataset]
            if idx >= len(rows):
                continue
            chosen.append(rows[idx])
            cursors[dataset] += 1
            if len(chosen) >= quota:
                break
            if cursors[dataset] < len(rows):
                next_active.append(dataset)
        active = next_active
    return chosen


def select_cases_for_transform(
    *,
    result_root: Path,
    transform: str,
    reasoning_quota: int,
    non_reasoning_quota: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    reasoning_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    non_reasoning_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: list[dict[str, Any]] = []
    for dataset in REASONING_DATASETS + NON_REASONING_DATASETS:
        rows = merged_improvements(result_root=result_root, transform=transform, dataset=dataset)
        all_rows.extend(rows)
        counts.append(
            {
                "transform": transform,
                "dataset": dataset,
                "group": dataset_group(dataset),
                "candidate_count": len(rows),
            }
        )
        if dataset_group(dataset) == "reasoning":
            reasoning_pool[dataset].extend(rows)
        else:
            non_reasoning_pool[dataset].extend(rows)

    selected = round_robin_pick(candidates_by_dataset=reasoning_pool, quota=reasoning_quota)
    selected.extend(round_robin_pick(candidates_by_dataset=non_reasoning_pool, quota=non_reasoning_quota))
    return selected, counts


def main() -> int:
    args = build_parser().parse_args()
    result_root = Path(args.result_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_counts: list[dict[str, Any]] = []
    selection_summary: list[dict[str, Any]] = []

    for transform in args.transforms:
        selected, counts = select_cases_for_transform(
            result_root=result_root,
            transform=transform,
            reasoning_quota=args.reasoning_quota,
            non_reasoning_quota=args.non_reasoning_quota,
        )
        all_counts.extend(counts)
        if len(selected) < args.per_transform_total:
            raise RuntimeError(
                f"Transform {transform} only has {len(selected)} selected cases, "
                f"below requested total {args.per_transform_total}."
            )
        selected = selected[: args.per_transform_total]
        manifest_path = output_dir / f"{transform}_manifest.json"
        manifest_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
        selection_summary.append(
            {
                "transform": transform,
                "manifest": str(manifest_path),
                "selected_count": len(selected),
                "reasoning_count": sum(1 for row in selected if row["group"] == "reasoning"),
                "non_reasoning_count": sum(1 for row in selected if row["group"] == "non_reasoning"),
            }
        )

    pd.DataFrame(all_counts).sort_values(["transform", "group", "dataset"]).to_csv(
        output_dir / "candidate_counts.csv",
        index=False,
    )
    (output_dir / "selection_summary.json").write_text(
        json.dumps(selection_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(selection_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
