import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception:
    pd = None


KNOWN_DATASET_ORDER = [
    "AI2D_TEST",
    "DynaMath",
    "MathVision",
    "MathVista_MINI",
    "OCRBench",
    "SEEDBench2_Plus",
    "VisuLogic",
    "LogicVista",
    "VisualPuzzles",
]

MODEL_DIR_NAME = "Qwen2VLChatReplay"

SCORE_SUFFIXES = [
    ("_gpt-4o_score.csv", "score_csv"),
    ("_gpt-4-turbo_score.csv", "score_csv"),
    ("_score.csv", "score_csv"),
    ("_acc.csv", "acc_csv"),
    ("_score.json", "score_json"),
]

ARTIFACT_SUFFIXES = [
    "_gpt-4o_score.csv",
    "_gpt-4-turbo_score.csv",
    "_score.csv",
    "_acc.csv",
    "_score.json",
    "_gpt-4o.xlsx.bak",
    "_gpt-4-turbo.xlsx.bak",
    "_gpt-4o.xlsx",
    "_gpt-4-turbo.xlsx",
    "_gpt-4o_result.xlsx",
    "_gpt-4-turbo_result.xlsx",
    "_gpt-4o.pkl",
    "_gpt-4-turbo.pkl",
    "_gpt-4o_result.pkl",
    "_gpt-4-turbo_result.pkl",
    "_gpt-4o_result.json",
    "_answer_format_report.json",
    "_answer_format_failures.jsonl",
    ".xlsx",
    ".xlsx.bak",
    ".tsv",
    ".pkl",
    ".json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "/path/to/vlmevalkit/runs/standard"
        ),
    )
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def split_task_name(task_name: str) -> Tuple[str, str]:
    parts = task_name.split("__")
    if len(parts) < 3:
        return task_name, task_name
    model = parts[0]
    setting = "__".join(parts[1:-1])
    return model, setting


def extract_dataset_name(file_name: str, model_dir_name: str) -> Optional[str]:
    prefix = f"{model_dir_name}_"
    if not file_name.startswith(prefix):
        return None
    rest = file_name[len(prefix):]
    for suffix in ARTIFACT_SUFFIXES:
        if rest.endswith(suffix):
            return rest[: -len(suffix)]
    return None


def parse_score_csv(path: Path) -> Tuple[Optional[str], Optional[float]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None, None
    if not rows:
        return None, None

    fieldnames = list(rows[0].keys())

    if "acc" in fieldnames:
        for key_col in ["Task&Skill", "Subject", "Setting", "split"]:
            if key_col in fieldnames:
                for row in rows:
                    if str(row.get(key_col, "")).strip() in {"Overall", "Average", "none"}:
                        try:
                            return "acc", float(row["acc"])
                        except Exception:
                            pass
        try:
            return "acc", float(rows[0]["acc"])
        except Exception:
            return None, None

    if "Overall" in fieldnames:
        try:
            return "Overall", float(rows[0]["Overall"])
        except Exception:
            return None, None

    for col in fieldnames:
        try:
            return col, float(rows[0][col])
        except Exception:
            continue
    return None, None


def parse_score_json(path: Path) -> Tuple[Optional[str], Optional[float]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, None

    for key in ["Final Score Norm", "Final Score", "Overall", "acc"]:
        if key in data and isinstance(data[key], (int, float)):
            return key, float(data[key])

    for key, value in data.items():
        if isinstance(value, (int, float)):
            return key, float(value)
    return None, None


def parse_score_file(path: Path, metric_type: str) -> Tuple[Optional[str], Optional[float]]:
    if metric_type in {"score_csv", "acc_csv"}:
        return parse_score_csv(path)
    if metric_type == "score_json":
        return parse_score_json(path)
    return None, None


def collect_from_dates(runs_root: Path, dates: List[str]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    best_scores: Dict[Tuple[str, str, str], Dict] = {}
    inventory: List[Dict] = []
    task_artifacts: Dict[Tuple[str, str, str], set] = {}
    task_score_datasets: Dict[Tuple[str, str, str], set] = {}

    for date_idx, date_tag in enumerate(dates):
        date_dir = runs_root / date_tag
        if not date_dir.exists():
            continue

        for pack_dir in sorted(date_dir.iterdir()):
            if not pack_dir.is_dir() or pack_dir.name.startswith("_"):
                continue

            for task_dir in sorted(pack_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                output_dir = task_dir / "output" / MODEL_DIR_NAME
                if not output_dir.is_dir():
                    continue

                model, setting = split_task_name(task_dir.name)
                task_key = (pack_dir.name, task_dir.name, model)

                for file_path in sorted(output_dir.iterdir()):
                    if not file_path.is_file():
                        continue
                    dataset = extract_dataset_name(file_path.name, MODEL_DIR_NAME)
                    if not dataset:
                        continue
                    task_artifacts.setdefault(task_key, set()).add(dataset)

                    metric_type = None
                    for suffix, mt in SCORE_SUFFIXES:
                        if file_path.name.endswith(suffix):
                            metric_type = mt
                            break
                    if metric_type is None:
                        continue

                    score_key, score_value = parse_score_file(file_path, metric_type)
                    inventory.append(
                        {
                            "date": date_tag,
                            "pack": pack_dir.name,
                            "task": task_dir.name,
                            "model": model,
                            "setting": setting,
                            "dataset": dataset,
                            "metric_type": metric_type,
                            "score_key": score_key,
                            "score_value": score_value,
                            "file_name": file_path.name,
                            "abs_path": str(file_path),
                        }
                    )
                    if score_value is None:
                        continue

                    task_score_datasets.setdefault(task_key, set()).add(dataset)
                    key = (pack_dir.name, task_dir.name, dataset)
                    candidate = {
                        "date": date_tag,
                        "date_idx": date_idx,
                        "pack": pack_dir.name,
                        "task": task_dir.name,
                        "model": model,
                        "setting": setting,
                        "dataset": dataset,
                        "metric_type": metric_type,
                        "score_key": score_key,
                        "score_value": score_value,
                        "file_name": file_path.name,
                        "abs_path": str(file_path),
                    }
                    prev = best_scores.get(key)
                    if prev is None or candidate["date_idx"] >= prev["date_idx"]:
                        best_scores[key] = candidate

    existing_primary_scores = sorted(
        [
            {k: v for k, v in record.items() if k != "date_idx"}
            for record in best_scores.values()
        ],
        key=lambda x: (x["pack"], x["task"], x["dataset"]),
    )

    missing_rows: List[Dict] = []
    for pack, task, model in sorted(task_artifacts.keys()):
        model, setting = split_task_name(task)
        datasets = task_artifacts[(pack, task, model)]
        scored = task_score_datasets.get((pack, task, model), set())
        for dataset in sorted(datasets):
            if dataset in scored:
                continue
            expected_metric_type = "unknown"
            if dataset == "OCRBench":
                expected_metric_type = "score_json"
            elif dataset in {"AI2D_TEST", "SEEDBench2_Plus"}:
                expected_metric_type = "acc_csv"
            else:
                expected_metric_type = "score_csv"
            missing_rows.append(
                {
                    "pack": pack,
                    "task": task,
                    "model": model,
                    "setting": setting,
                    "dataset": dataset,
                    "expected_metric_type": expected_metric_type,
                    "task_dir": str(runs_root / "*" / pack / task / "output" / MODEL_DIR_NAME),
                }
            )

    missing_rows = sorted(missing_rows, key=lambda x: (x["pack"], x["task"], x["dataset"]))
    inventory = sorted(inventory, key=lambda x: (x["pack"], x["task"], x["dataset"], x["file_name"]))
    return existing_primary_scores, missing_rows, inventory


def dataset_order(values: List[str]) -> List[str]:
    known = [d for d in KNOWN_DATASET_ORDER if d in values]
    extras = sorted([d for d in values if d not in KNOWN_DATASET_ORDER])
    return known + extras


def compute_task_avg_output_lengths(existing_scores: List[Dict]) -> Dict[str, float]:
    if pd is None:
        return {}

    grouped: Dict[str, List[Dict]] = {}
    for row in existing_scores:
        label = f'{row["pack"]}|{row["model"]}|{row["setting"]}'
        grouped.setdefault(label, []).append(row)

    result: Dict[str, float] = {}
    for label, records in grouped.items():
        total_chars = 0
        total_count = 0
        for row in records:
            score_path = Path(row["abs_path"])
            model_dir = score_path.parent
            dataset = row["dataset"]
            infer_path = model_dir / f"{MODEL_DIR_NAME}_{dataset}.xlsx"
            if not infer_path.exists():
                infer_path = model_dir / f"{MODEL_DIR_NAME}_{dataset}.tsv"
            if not infer_path.exists() or infer_path.stat().st_size == 0:
                continue
            try:
                if infer_path.suffix == ".tsv":
                    df = pd.read_csv(infer_path, sep="\t")
                else:
                    df = pd.read_excel(infer_path)
            except Exception:
                continue
            output_col = None
            for col in ["prediction", "detailed_prediction", "description"]:
                if col in df.columns:
                    output_col = col
                    break
            if output_col is None:
                continue
            for value in df[output_col].tolist():
                if value is None:
                    continue
                try:
                    if pd.isna(value):
                        continue
                except Exception:
                    pass
                text = str(value).strip()
                if not text:
                    continue
                total_chars += len(text)
                total_count += 1
        if total_count > 0:
            result[label] = total_chars / total_count
    return result


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_pivots(existing_scores: List[Dict], output_dir: Path) -> None:
    pivot_dir = output_dir / "pivot_by_setting"
    pivot_dir.mkdir(parents=True, exist_ok=True)

    if not existing_scores:
        return

    all_datasets = dataset_order(sorted({row["dataset"] for row in existing_scores}))
    avg_output_lengths = compute_task_avg_output_lengths(existing_scores)

    all_map: Dict[str, Dict[str, float]] = {}
    for row in existing_scores:
        label = f'{row["pack"]}|{row["model"]}|{row["setting"]}'
        all_map.setdefault(label, {})
        all_map[label][row["dataset"]] = row["score_value"]

    all_rows = []
    for setting_label in sorted(all_map.keys()):
        row = {
            "setting": setting_label,
            "avg_output_length": avg_output_lengths.get(setting_label, ""),
        }
        for dataset in all_datasets:
            row[dataset] = all_map[setting_label].get(dataset, "")
        all_rows.append(row)
    write_csv(
        pivot_dir / "ALL__dataset_x_setting.csv",
        all_rows,
        ["setting", "avg_output_length"] + all_datasets,
    )

    grouped: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    for row in existing_scores:
        key = (row["pack"], row["model"])
        grouped.setdefault(key, {})
        grouped[key].setdefault(row["setting"], {})
        grouped[key][row["setting"]][row["dataset"]] = row["score_value"]

    for (pack, model), setting_map in sorted(grouped.items()):
        sub_datasets = dataset_order(sorted({d for values in setting_map.values() for d in values.keys()}))
        rows = []
        for setting in sorted(setting_map.keys()):
            full_label = f"{pack}|{model}|{setting}"
            row = {
                "setting": setting,
                "avg_output_length": avg_output_lengths.get(full_label, ""),
            }
            for dataset in sub_datasets:
                row[dataset] = setting_map[setting].get(dataset, "")
            rows.append(row)
        out_name = f"{pack}__{model}__dataset_x_setting.csv"
        write_csv(pivot_dir / out_name, rows, ["setting", "avg_output_length"] + sub_datasets)


def main() -> None:
    args = parse_args()
    existing_scores, missing_rows, inventory = collect_from_dates(args.runs_root, args.dates)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        args.output_dir / "existing_primary_scores.csv",
        existing_scores,
        ["date", "pack", "task", "model", "setting", "dataset", "metric_type", "score_key", "score_value", "file_name", "abs_path"],
    )
    write_csv(
        args.output_dir / "missing_results_only.csv",
        missing_rows,
        ["pack", "task", "model", "setting", "dataset", "expected_metric_type", "task_dir"],
    )
    write_csv(
        args.output_dir / "result_file_inventory.csv",
        inventory,
        ["date", "pack", "task", "model", "setting", "dataset", "metric_type", "score_key", "score_value", "file_name", "abs_path"],
    )

    write_pivots(existing_scores, args.output_dir)


if __name__ == "__main__":
    main()
