#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from vlmeval.dataset import build_dataset
from vlmeval.smp.file import LMUDataRoot, localize_df
from vlmeval.smp.vlm import decode_base64_to_image_file


DEFAULT_REASONING = ["DynaMath", "LogicVista", "MathVision", "VisualPuzzles"]
DEFAULT_NON_REASONING = ["AI2D_TEST", "OCRBench"]

OPTION_RE = re.compile(r"\b([ABCD])\b", flags=re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine IQI-correct / IIQ-wrong Qwen2.5-VL-32B cases with localized image paths."
    )
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reasoning-datasets", nargs="+", default=DEFAULT_REASONING)
    parser.add_argument("--non-reasoning-datasets", nargs="+", default=DEFAULT_NON_REASONING)
    return parser


def resolve_eval_file(task_root: Path, dataset: str) -> Path:
    candidates = [
        task_root / f"Qwen2VLChatReplay_{dataset}_gpt-4o-mini.xlsx",
        task_root / f"Qwen2VLChatReplay_{dataset}_gpt4o-mini.xlsx",
        task_root / f"Qwen2VLChatReplay_{dataset}.xlsx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No eval/infer file found for dataset={dataset} under {task_root}")


def load_eval_df(result_root: Path, mode: str, dataset: str) -> pd.DataFrame:
    task_root = result_root / "default" / mode / "baseline" / "qwen25vl_32b" / "Qwen2VLChatReplay"
    return pd.read_excel(resolve_eval_file(task_root, dataset))


def normalize_mcq_letter(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    match = OPTION_RE.search(text.upper())
    return match.group(1).upper() if match else ""


def normalize_visual_puzzles(df: pd.DataFrame) -> pd.Series:
    pred = df.get("extracted_answer", pd.Series([""] * len(df))).fillna("").astype(str).str.strip().str.upper()
    missing = pred == ""
    if missing.any():
        pred = pred.copy()
        pred.loc[missing] = df.loc[missing, "prediction"].map(normalize_mcq_letter)
    gold = df["answer"].fillna("").astype(str).str.strip().str.upper()
    return pred == gold


def normalize_ocrbench(df: pd.DataFrame) -> pd.Series:
    preds = df["prediction"].fillna("").astype(str)
    answers = df["answer"].fillna("").astype(str)

    def row_correct(pred: str, answer_text: str, category: str) -> bool:
        pred_norm = pred.strip().replace("\n", " ")
        try:
            answers_list = ast.literal_eval(answer_text)
            if not isinstance(answers_list, list):
                answers_list = [str(answers_list)]
        except Exception:
            answers_list = [answer_text]
        if category == "Handwritten Mathematical Expression Recognition":
            pred_cmp = pred_norm.replace(" ", "")
            return any(str(ans).strip().replace("\n", " ").replace(" ", "") in pred_cmp for ans in answers_list)
        pred_cmp = pred_norm.lower()
        return any(str(ans).lower().strip().replace("\n", " ") in pred_cmp for ans in answers_list)

    categories = df.get("category", pd.Series([""] * len(df))).fillna("").astype(str)
    return pd.Series(
        [row_correct(pred, answer, category) for pred, answer, category in zip(preds, answers, categories)],
        index=df.index,
    )


def normalize_correct(dataset: str, df: pd.DataFrame) -> pd.Series:
    if dataset == "DynaMath":
        return df["correct"].fillna(False).astype(bool)
    if dataset == "LogicVista":
        return df["hit"].fillna(0).astype(int).astype(bool)
    if dataset == "MathVision":
        return df["hit_score"].fillna(0).astype(float) > 0.5
    if dataset == "VisualPuzzles":
        return normalize_visual_puzzles(df)
    if dataset in {"AI2D_TEST", "SEEDBench2_Plus"}:
        pred = df["prediction"].map(normalize_mcq_letter)
        gold = df["answer"].fillna("").astype(str).str.strip().str.upper()
        return pred == gold
    if dataset == "OCRBench":
        return normalize_ocrbench(df)
    raise KeyError(f"Unsupported dataset for correctness normalization: {dataset}")


def dataset_group(dataset: str, reasoning: list[str], non_reasoning: list[str]) -> str:
    if dataset in reasoning:
        return "reasoning"
    if dataset in non_reasoning:
        return "non_reasoning"
    raise KeyError(f"Unsupported dataset group for {dataset}")


def localize_dataset_df(dataset: str) -> pd.DataFrame:
    ds = build_dataset(dataset)
    if ds is None:
        raise RuntimeError(f"Failed to build dataset: {dataset}")
    data = ds.data.copy()
    if "image_path" not in data.columns:
        if "image" not in data.columns:
            raise KeyError(f"Dataset {dataset} has neither image nor image_path columns")
        data = localize_df(data, dataset, nproc=8)
    return data


def image_stats(image_path: str) -> dict[str, float]:
    image = Image.open(image_path).convert("L")
    arr = np.asarray(image, dtype=np.float32)
    hist = np.histogram(arr, bins=256, range=(0, 255), density=True)[0]
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log2(hist)).sum())
    gy, gx = np.gradient(arr / 255.0)
    grad = np.sqrt(gx * gx + gy * gy)
    edge_density = float((grad > 0.08).mean())
    return {
        "image_width": float(arr.shape[1]),
        "image_height": float(arr.shape[0]),
        "grayscale_entropy": entropy,
        "edge_density": edge_density,
        "pixel_std": float(arr.std()),
    }


def resolve_image_path(dataset: str, image_path: str, image_blob: Any, source_index: int) -> str:
    raw = Path(str(image_path))
    candidates = [
        raw,
        Path(LMUDataRoot()) / "images" / dataset / raw,
        Path(LMUDataRoot()) / "images" / dataset / raw.name,
        Path(LMUDataRoot()) / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    if isinstance(image_blob, str) and len(image_blob) > 64:
        fallback = Path(LMUDataRoot()) / "images" / dataset / "__task4_cache__" / f"{source_index}.jpg"
        decode_base64_to_image_file(image_blob, str(fallback))
        if fallback.exists():
            return str(fallback)
    raise FileNotFoundError(f"Unable to resolve image_path for dataset={dataset}: {image_path}")


def main() -> int:
    args = build_parser().parse_args()
    result_root = Path(args.result_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []

    datasets = list(args.reasoning_datasets) + list(args.non_reasoning_datasets)
    for dataset in datasets:
        group = dataset_group(dataset, args.reasoning_datasets, args.non_reasoning_datasets)
        data = localize_dataset_df(dataset)
        keep_cols = ["index", "image_path", "question", "answer"]
        if "image" in data.columns:
            keep_cols.append("image")
        for col in ["category", "subcategory", "subject", "skill", "broad_capability", "specific_capability", "difficulty", "knowledge_level", "reasoning"]:
            if col in data.columns and col not in keep_cols:
                keep_cols.append(col)
        data = data[keep_cols].copy()

        iqi = load_eval_df(result_root, "image_text_image", dataset).copy()
        iiq = load_eval_df(result_root, "image_image_text", dataset).copy()
        iqi["_correct"] = normalize_correct(dataset, iqi)
        iiq["_correct"] = normalize_correct(dataset, iiq)

        merged = (
            data.merge(iqi[["index", "_correct", "prediction"]], on="index", how="inner")
            .rename(columns={"_correct": "iqi_correct", "prediction": "iqi_prediction"})
            .merge(iiq[["index", "_correct", "prediction"]], on="index", how="inner")
            .rename(columns={"_correct": "iiq_correct", "prediction": "iiq_prediction"})
        )
        merged = merged[(merged["iqi_correct"]) & (~merged["iiq_correct"])].copy()
        merged = merged.sort_values("index").reset_index(drop=True)

        count_rows.append(
            {
                "dataset": dataset,
                "group": group,
                "candidate_count": int(len(merged)),
            }
        )

        for _, row in merged.iterrows():
            resolved_image_path = resolve_image_path(
                dataset,
                str(row["image_path"]),
                row.get("image", None),
                int(row["index"]),
            )
            record = {
                "candidate_id": f"{dataset}__idx{int(row['index'])}",
                "source_dataset": dataset,
                "group": group,
                "source_index": int(row["index"]),
                "image_path": resolved_image_path,
                "image_path_raw": str(row["image_path"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "iqi_prediction": str(row["iqi_prediction"]),
                "iiq_prediction": str(row["iiq_prediction"]),
            }
            for col in ["category", "subcategory", "subject", "skill", "broad_capability", "specific_capability", "difficulty", "knowledge_level", "reasoning"]:
                if col in row:
                    record[col] = row[col]
            record.update(image_stats(resolved_image_path))
            candidate_rows.append(record)

    candidates = pd.DataFrame(candidate_rows)
    counts = pd.DataFrame(count_rows).sort_values(["group", "dataset"]).reset_index(drop=True)
    counts.to_csv(output_dir / "candidate_counts.csv", index=False)
    candidates.to_csv(output_dir / "candidates.csv", index=False)

    summary = {
        "result_root": str(result_root),
        "reasoning_datasets": list(args.reasoning_datasets),
        "non_reasoning_datasets": list(args.non_reasoning_datasets),
        "candidate_counts": counts.to_dict(orient="records"),
        "total_candidates": int(len(candidates)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
