#!/usr/bin/env python3
"""
从 runs/standard/20260306 和 20260307 收集结果，更新
ALL__dataset_x_setting_no_reasoning.csv 和 ALL__dataset_x_setting_reasoning.csv。

数据源规则（按 pack 区分）：
- no_reasoning 表：只使用 pack 名含 "direct" 的跑法结果（直接作答，无推理）。
- reasoning 表：只使用 pack 名含 "default" 或 "default_prompt" 的跑法结果（带推理 prompt）。
  - 前 6 个 dataset 来自 existing_primary_scores（均为 default_prompt）及 default 类 run。
  - VisuLogic / LogicVista / VisualPuzzles 仅来自 three_sets 的 default 跑法，且只有 72B 有
    three_sets default（three_sets_qwen25_72b_default_*），故这三列在 reasoning 表里只填 72B。
"""
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs" / "standard"
DATES = ["20260306", "20260307"]
NO_REASONING_CSV = Path(__file__).resolve().parents[1] / "ALL__dataset_x_setting_no_reasoning.csv"
REASONING_CSV = Path(__file__).resolve().parents[1] / "ALL__dataset_x_setting_reasoning.csv"

# 与 summary 一致：只关心前 9 个 dataset 列（不含 Replay*）
DATASET_COLS = [
    "AI2D_TEST", "DynaMath", "MathVision", "MathVista_MINI",
    "OCRBench", "SEEDBench2_Plus", "VisuLogic", "LogicVista", "VisualPuzzles",
]
THREE_SPECIAL_DATASETS = {"VisuLogic", "LogicVista", "VisualPuzzles"}
REASONING_72B_ONLY_MODEL = "Qwen2.5-VL-72B-Instruct"

# 可能的 output 子目录名（不同模型）
OUTPUT_SUBDIRS = ["Qwen2VLChatReplay", "MiniCPM-V-4_5-Replay"]


def split_task_name(task_name: str) -> Tuple[str, str]:
    """Task 目录名: Model__setting__last1 -> (model, setting)"""
    parts = task_name.split("__")
    if len(parts) < 3:
        return task_name, ""
    model = parts[0]
    setting = "__".join(parts[1:-1])
    return model, setting


def parse_acc_csv(path: Path) -> Optional[float]:
    """解析 *_acc.csv：取 Overall 行的 acc（百分比）或 Overall 列（0-1）。"""
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    if not rows:
        return None
    fn = list(rows[0].keys())
    # 格式 1: category,tot,hit,acc -> 找 category=Overall 的 acc
    if "acc" in fn and ("category" in fn or "split" in fn or "group" in fn):
        if "category" in fn:
            key_col = "category"
        elif "split" in fn:
            key_col = "split"
        else:
            key_col = "group"
        for row in rows:
            key_val = str(row.get(key_col, "")).strip().lower()
            if key_val in ("overall", "none", "average"):
                try:
                    return float(row["acc"])
                except Exception:
                    pass
        try:
            return float(rows[0]["acc"])
        except Exception:
            pass
    # 格式 2: split,Overall,... -> 第一行 Overall 列是 0-1
    if "Overall" in fn:
        try:
            v = float(rows[0]["Overall"])
            return v * 100.0 if v <= 1.0 else v
        except Exception:
            pass
    return None


def parse_score_csv(path: Path) -> Optional[float]:
    """解析 *_score.csv：取 acc 或 Overall（0-1 转百分比）。"""
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    if not rows:
        return None
    fn = list(rows[0].keys())
    if "acc" in fn:
        for key_col in ["Task&Skill", "Subject", "Setting", "split"]:
            if key_col in fn:
                for row in rows:
                    key_val = str(row.get(key_col, "")).strip().lower()
                    if key_val in ("overall", "average", "none"):
                        try:
                            return float(row["acc"])
                        except Exception:
                            pass
        try:
            return float(rows[0]["acc"])
        except Exception:
            pass
    if "Overall" in fn:
        try:
            v = float(rows[0]["Overall"])
            return v * 100.0 if v <= 1.0 else v
        except Exception:
            pass
    return None


def parse_score_json(path: Path) -> Optional[float]:
    """解析 *_score.json：取 Final Score Norm / Overall / acc。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ["Final Score Norm", "Final Score", "Overall", "acc"]:
        if key in data and isinstance(data[key], (int, float)):
            return float(data[key])
    return None


def extract_dataset_from_filename(filename: str, prefix: str) -> Optional[str]:
    """从文件名提取 dataset：prefix_DataSetName_suffix -> DataSetName"""
    if not filename.startswith(prefix + "_"):
        return None
    rest = filename[len(prefix) + 1 :]
    for suffix in ["_gpt-4o_acc.csv", "_gpt-4o_score.csv", "_acc.csv", "_score.csv", "_score.json"]:
        if rest.endswith(suffix):
            return rest[: -len(suffix)]
    return None


def _is_reasoning_pack(pack_name: str) -> bool:
    """Pack 名含 default 或 default_prompt 视为 reasoning（带推理）跑法。"""
    return "default" in pack_name.lower() or "default_prompt" in pack_name.lower()


def _is_direct_pack(pack_name: str) -> bool:
    """Pack 名含 direct 视为 no_reasoning（直接作答）跑法。"""
    return "direct" in pack_name.lower()


def collect_scores_from_runs() -> Tuple[
    Dict[Tuple[str, str], Dict[str, Tuple[str, float]]],
    Dict[Tuple[str, str], Dict[str, Tuple[str, float]]],
]:
    """
    从 20260306 和 20260307 扫描所有 score 文件，按 pack 类型拆成两份。
    返回:
      no_reasoning: 仅来自 pack 含 "direct" 的结果。
      reasoning: 仅来自 pack 含 "default"/"default_prompt" 的结果；且 VisuLogic/LogicVista/VisualPuzzles
                 只接受 model==72B（three_sets 下只有 72B 有 default 跑法）。
    """
    no_res: Dict[Tuple[str, str], Dict[str, Tuple[str, float]]] = {}
    res: Dict[Tuple[str, str], Dict[str, Tuple[str, float]]] = {}

    for date in DATES:
        date_dir = RUNS_ROOT / date
        if not date_dir.exists():
            continue
        for pack_dir in sorted(date_dir.iterdir()):
            if not pack_dir.is_dir() or pack_dir.name.startswith("_"):
                continue
            pack_name = pack_dir.name
            is_direct = _is_direct_pack(pack_name)
            is_reasoning = _is_reasoning_pack(pack_name)
            if not is_direct and not is_reasoning:
                continue
            for task_dir in sorted(pack_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                model, setting = split_task_name(task_dir.name)
                output_dir = task_dir / "output"
                if not output_dir.is_dir():
                    continue
                for subdir_name in OUTPUT_SUBDIRS:
                    out_sub = output_dir / subdir_name
                    if not out_sub.is_dir():
                        continue
                    prefix = subdir_name
                    for fpath in sorted(out_sub.iterdir()):
                        if not fpath.is_file():
                            continue
                        dataset = extract_dataset_from_filename(fpath.name, prefix)
                        if not dataset or dataset not in DATASET_COLS:
                            continue
                        value = None
                        if fpath.suffix == ".json" and "_score" in fpath.name:
                            value = parse_score_json(fpath)
                        elif fpath.name.endswith("_acc.csv") or fpath.name.endswith("_gpt-4o_acc.csv"):
                            value = parse_acc_csv(fpath)
                        elif fpath.name.endswith("_score.csv") or fpath.name.endswith("_gpt-4o_score.csv"):
                            value = parse_score_csv(fpath)
                        if value is None:
                            continue
                        val = round(value, 2)
                        key = (model, setting)
                        # no_reasoning: 只收 direct
                        if is_direct:
                            no_res.setdefault(key, {})
                            ex = no_res[key].get(dataset)
                            if ex is None or date >= ex[0]:
                                no_res[key][dataset] = (date, val)
                        # reasoning: 只收 default；且三个特殊数据集只收 72B
                        if is_reasoning:
                            if dataset in THREE_SPECIAL_DATASETS and model != REASONING_72B_ONLY_MODEL:
                                continue
                            res.setdefault(key, {})
                            ex = res[key].get(dataset)
                            if ex is None or date >= ex[0]:
                                res[key][dataset] = (date, val)
                    break
    return no_res, res


def score_from_primary_scores_csv() -> Dict[Tuple[str, str], Dict[str, Tuple[str, float]]]:
    """从已有 existing_primary_scores 表得到 (model, setting) -> { dataset: (date, value_percent) }"""
    summary_path = RUNS_ROOT / "20260307" / "_summary_csv_from_20260306_20260307" / "existing_primary_scores.csv"
    if not summary_path.exists():
        return {}
    result: Dict[Tuple[str, str], Dict[str, Tuple[str, float]]] = {}
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("model", "").strip()
            setting = row.get("setting", "").strip()
            dataset = row.get("dataset", "").strip()
            if not model or dataset not in DATASET_COLS:
                continue
            try:
                raw = float(row.get("score_value", 0))
            except Exception:
                continue
            sk = (row.get("score_key") or "").strip()
            # acc_csv Overall 为 0-1；score_csv Overall 为 0-1；score_csv acc / score_json 已是百分比
            if sk == "Overall" and raw <= 1.0:
                value = round(raw * 100.0, 2)
            else:
                value = round(raw, 2)
            date = (row.get("date") or "").strip()
            key = (model, setting)
            result.setdefault(key, {})
            if dataset not in result[key] or date >= result[key][dataset][0]:
                result[key][dataset] = (date, value)
    return result


def merge_scores(
    from_runs: Dict[Tuple[str, str], Dict[str, Tuple[str, float]]],
    from_csv: Dict[Tuple[str, str], Dict[str, Tuple[str, float]]],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """合并 reasoning 两源（primary_scores CSV + runs 中 default 类），同 key 优先日期更新。"""
    merged: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, d in from_csv.items():
        merged.setdefault(key, {})
        for ds, (date, val) in d.items():
            merged[key][ds] = val
    for key, d in from_runs.items():
        merged.setdefault(key, {})
        for ds, (date, val) in d.items():
            existing_date = from_csv.get(key, {}).get(ds, (None, None))[0]
            if existing_date is None or date >= existing_date:
                merged[key][ds] = val
    return merged


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    """返回 (fieldnames, rows)。"""
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fn = reader.fieldnames or []
        rows = list(reader)
    return fn, rows


def is_data_row(setting_cell: str) -> bool:
    """第一列是否形如 Model|setting（数据行，非表头/空行）。"""
    if not setting_cell or "|" not in setting_cell:
        return False
    # 排除 pack|model|setting 三段的（replay 段）
    parts = setting_cell.split("|")
    return len(parts) == 2 and all(p.strip() for p in parts)


def clear_reasoning_three_special_for_non72b(path: Path, setting_col: str = "setting") -> int:
    """
    将 reasoning 表中 VisuLogic/LogicVista/VisualPuzzles 三列里「非 72B」的行清空（纠正误填）。
    返回清空的格子数。
    """
    fn, rows = read_csv_rows(path)
    if not fn:
        return 0
    if setting_col not in fn:
        setting_col = fn[0] if fn else "setting"
    cleared = 0
    for row in rows:
        raw_setting = row.get(setting_col, "")
        if not is_data_row(raw_setting):
            continue
        parts = [p.strip() for p in raw_setting.split("|")]
        if len(parts) != 2:
            continue
        model = parts[0]
        if model == REASONING_72B_ONLY_MODEL:
            continue
        for col in THREE_SPECIAL_DATASETS:
            if col not in fn:
                continue
            current = (row.get(col) or "").strip()
            if not current or current == "0.00":
                continue
            row[col] = ""
            cleared += 1
    if cleared:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fn)
            writer.writeheader()
            writer.writerows(rows)
    return cleared


def update_csv(
    path: Path,
    score_map: Dict[Tuple[str, str], Dict[str, float]],
    dataset_columns: List[str],
    setting_col: str = "setting",
) -> None:
    """
    只更新当前为空的格子；若 score_map 中有该 (model, setting) 的 dataset 值则填入。
    保留原有数字格式（两位小数，带空格可保留）。
    """
    fn, rows = read_csv_rows(path)
    if not fn:
        return
    if setting_col not in fn:
        setting_col = fn[0] if fn else "setting"
    updated = 0
    for row in rows:
        raw_setting = row.get(setting_col, "")
        if not is_data_row(raw_setting):
            continue
        parts = [p.strip() for p in raw_setting.split("|")]
        if len(parts) != 2:
            continue
        model, setting = parts[0], parts[1]
        key = (model, setting)
        if key not in score_map:
            continue
        for col in dataset_columns:
            if col not in fn:
                continue
            current = (row.get(col) or "").strip()
            val = score_map[key].get(col)
            if val is None:
                continue
            formatted = f"{val:.2f}"
            if current == formatted:
                continue
            row[col] = f"{formatted} "
            updated += 1
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fn)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated {path.name}: {updated} cells updated.")


def main() -> None:
    from_runs_no_res, from_runs_res = collect_scores_from_runs()
    from_csv = score_from_primary_scores_csv()
    # no_reasoning 表：仅用 direct 跑法，不用 primary_scores（那是 default_prompt）
    score_map_no_reasoning = {k: {ds: v[1] for ds, v in d.items()} for k, d in from_runs_no_res.items()}
    # reasoning 表：primary_scores（default_prompt 前 6 个 dataset）+ runs 中 default 类（含 72B 三特殊）
    score_map_reasoning = merge_scores(from_runs_res, from_csv)
    print(f"No-reasoning: {len(score_map_no_reasoning)} (model, setting) keys from direct packs.")
    print(f"Reasoning: {len(score_map_reasoning)} (model, setting) keys from default/primary_scores.")

    if NO_REASONING_CSV.exists():
        update_csv(NO_REASONING_CSV, score_map_no_reasoning, DATASET_COLS)
    else:
        print(f"Not found: {NO_REASONING_CSV}")

    if REASONING_CSV.exists():
        cleared = clear_reasoning_three_special_for_non72b(REASONING_CSV)
        if cleared:
            print(f"Cleared {cleared} wrong cells (non-72B VisuLogic/LogicVista/VisualPuzzles) in reasoning.")
        update_csv(REASONING_CSV, score_map_reasoning, DATASET_COLS)
    else:
        print(f"Not found: {REASONING_CSV}")


if __name__ == "__main__":
    main()
