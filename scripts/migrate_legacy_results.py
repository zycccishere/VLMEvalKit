#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required to run scripts/migrate_legacy_results.py") from exc


TABLE_MODEL_TO_KEY = {
    "Qwen2 7B": "qwen2vl_7b",
    "Qwen2.5 7B": "qwen25vl_7b",
    "Qwen3.5 4B": "qwen35_4b",
    "Qwen3.5 9B": "qwen35_9b",
    "Qwen3.5 27B": "qwen35_27b",
    "Qwen3.5 35B-A3B": "qwen35_35b_a3b",
    "MiniCPM-V": "minicpm_v_45",
    "MiniCPM-o": "minicpm_o_45",
}

LEGACY_MODEL_PATTERNS = [
    ("Qwen3.5-35B-A3B", "qwen35_35b_a3b"),
    ("Qwen3.5-27B", "qwen35_27b"),
    ("Qwen3.5-9B", "qwen35_9b"),
    ("Qwen3.5-4B", "qwen35_4b"),
    ("Qwen2.5-VL-7B", "qwen25vl_7b"),
    ("Qwen2-VL-7B", "qwen2vl_7b"),
    ("MiniCPM-o-4_5", "minicpm_o_45"),
    ("MiniCPM-V-4_5", "minicpm_v_45"),
]


@dataclass
class MarkerRecord:
    model_key: str
    registry_name: str
    policy_key: str
    mode: str
    dataset: str
    overall: float
    source: str
    provenance: str
    priority: int


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def to_float(value: Any) -> float | None:
    if is_blank(value):
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def normalize_metric(value: float) -> float:
    if 0.0 <= value <= 1.0:
        return value * 100.0
    return value


def canonical_mode(value: str) -> str:
    mode = value.strip()
    if mode == "none":
        return "image_text"
    return mode


def infer_policy_from_path(path: Path) -> str | None:
    text = str(path).lower()
    if "__directly_answer__" in text or "direct_answer" in text or "no_reasoning" in text:
        return "direct"
    if "__identity__" in text or "default_prompt" in text or "reasoning" in text:
        return "default"
    return None


def infer_mode_from_path(path: Path, valid_modes: list[str]) -> str | None:
    text = str(path)
    candidates = ["none"] + sorted(valid_modes, key=len, reverse=True)
    for mode in candidates:
        if f"__{mode}__" in text:
            return canonical_mode(mode)
    return None


def infer_model_key_from_path(path: Path, active_model_keys: set[str]) -> str | None:
    text = str(path)
    for pattern, model_key in LEGACY_MODEL_PATTERNS:
        if model_key in active_model_keys and pattern in text:
            return model_key
    return None


def infer_dataset_from_name(name: str, valid_datasets: list[str]) -> str | None:
    for dataset in sorted(valid_datasets, key=len, reverse=True):
        if f"_{dataset}_" in name or name.endswith(f"_{dataset}.csv"):
            return dataset
    return None


def parse_overall_from_csv(path: Path) -> float | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except Exception:
        return None
    if not rows:
        return None

    def row_label(row: dict[str, Any]) -> str:
        for key in ("group", "split", "setting", "type"):
            value = row.get(key)
            if not is_blank(value):
                return str(value).strip().lower()
        return ""

    preferred_numeric_keys = ["Overall", "acc", "overall", "score", "value"]
    preferred_labels = {"overall", "average", "none"}

    for row in rows:
        label = row_label(row)
        if label not in preferred_labels:
            continue
        for key in preferred_numeric_keys:
            value = to_float(row.get(key))
            if value is not None:
                return normalize_metric(value)

    for row in rows:
        for key in preferred_numeric_keys:
            value = to_float(row.get(key))
            if value is not None:
                return normalize_metric(value)
    return None


def marker_path(results_root: Path, record: MarkerRecord) -> Path:
    return (
        results_root
        / record.policy_key
        / record.mode
        / record.model_key
        / record.registry_name
        / f"{record.registry_name}_{record.dataset}_acc.csv"
    )


def write_marker(results_root: Path, record: MarkerRecord) -> Path:
    path = marker_path(results_root, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "dataset",
                "overall",
                "source",
                "provenance",
                "model_key",
                "registry_name",
                "policy",
                "mode",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": record.dataset,
                "overall": f"{record.overall:.10f}".rstrip("0").rstrip("."),
                "source": record.source,
                "provenance": record.provenance,
                "model_key": record.model_key,
                "registry_name": record.registry_name,
                "policy": record.policy_key,
                "mode": record.mode,
            }
        )
    return path


def load_active_registry(models_cfg: dict[str, Any], matrix_cfg: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for model_key in matrix_cfg["models"]:
        if model_key in models_cfg["models"]:
            out[model_key] = str(models_cfg["models"][model_key]["registry_name"])
    return out


def load_table_records(
    path: Path,
    policy_key: str,
    registry_names: dict[str, str],
    valid_modes: set[str],
    valid_datasets: set[str],
) -> list[MarkerRecord]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]]
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: list[MarkerRecord] = []
    for row in rows:
        model_key = TABLE_MODEL_TO_KEY.get(str(row.get("Model", "")).strip())
        if model_key is None or model_key not in registry_names:
            continue
        mode = canonical_mode(str(row.get("setting", "")).strip())
        if mode not in valid_modes:
            continue
        for dataset, raw in row.items():
            if dataset in {"Model", "setting"} or dataset not in valid_datasets or is_blank(raw):
                continue
            value = to_float(raw)
            if value is None:
                continue
            out.append(
                MarkerRecord(
                    model_key=model_key,
                    registry_name=registry_names[model_key],
                    policy_key=policy_key,
                    mode=mode,
                    dataset=dataset,
                    overall=normalize_metric(value),
                    source="table",
                    provenance=str(path),
                    priority=1,
                )
            )
    return out


def scan_legacy_records(
    roots: list[Path],
    registry_names: dict[str, str],
    valid_modes: list[str],
    valid_datasets: list[str],
) -> list[MarkerRecord]:
    active_model_keys = set(registry_names)
    out: list[MarkerRecord] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            if not path.name.endswith(("_acc.csv", "_score.csv")):
                continue
            model_key = infer_model_key_from_path(path, active_model_keys)
            if model_key is None:
                continue
            policy_key = infer_policy_from_path(path)
            if policy_key is None:
                continue
            mode = infer_mode_from_path(path, valid_modes)
            if mode is None:
                continue
            dataset = infer_dataset_from_name(path.name, valid_datasets)
            if dataset is None:
                continue
            overall = parse_overall_from_csv(path)
            if overall is None:
                continue
            out.append(
                MarkerRecord(
                    model_key=model_key,
                    registry_name=registry_names[model_key],
                    policy_key=policy_key,
                    mode=mode,
                    dataset=dataset,
                    overall=overall,
                    source="legacy",
                    provenance=str(path),
                    priority=2,
                )
            )
    return out


def merge_records(records: list[MarkerRecord]) -> dict[tuple[str, str, str, str], MarkerRecord]:
    merged: dict[tuple[str, str, str, str], MarkerRecord] = {}
    for record in records:
        key = (record.model_key, record.policy_key, record.mode, record.dataset)
        current = merged.get(key)
        if current is None:
            merged[key] = record
            continue
        if record.priority > current.priority:
            merged[key] = record
            continue
    return merged


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Migrate legacy/table results into by-setting acc markers.")
    parser.add_argument("--matrix-config", type=Path, default=script_dir / "configs" / "matrix.yaml")
    parser.add_argument("--model-config", type=Path, default=script_dir / "configs" / "models.yaml")
    parser.add_argument("--direct-table", type=Path, required=True)
    parser.add_argument("--default-table", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, action="append", default=[])
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix_cfg = load_yaml(args.matrix_config)
    models_cfg = load_yaml(args.model_config)
    repo_root = Path(str(matrix_cfg["repo_root"]))
    results_root = Path(str(matrix_cfg["results_root"]))
    if not results_root.is_absolute():
        results_root = repo_root / results_root

    registry_names = load_active_registry(models_cfg, matrix_cfg)
    valid_modes = [canonical_mode(str(mode)) for mode in matrix_cfg["replay_modes"]]
    valid_mode_set = set(valid_modes)
    valid_datasets = [str(dataset) for dataset in matrix_cfg["datasets"]]
    valid_dataset_set = set(valid_datasets)

    all_records: list[MarkerRecord] = []
    all_records.extend(load_table_records(args.direct_table, "direct", registry_names, valid_mode_set, valid_dataset_set))
    all_records.extend(load_table_records(args.default_table, "default", registry_names, valid_mode_set, valid_dataset_set))
    all_records.extend(scan_legacy_records(args.legacy_root, registry_names, valid_modes, valid_datasets))

    merged = merge_records(all_records)
    written = []
    source_counts: dict[str, int] = {}
    for record in merged.values():
        path = write_marker(results_root, record)
        source_counts[record.source] = source_counts.get(record.source, 0) + 1
        written.append(
            {
                "path": str(path),
                "source": record.source,
                "provenance": record.provenance,
                "model_key": record.model_key,
                "policy": record.policy_key,
                "mode": record.mode,
                "dataset": record.dataset,
                "overall": record.overall,
            }
        )

    summary = {
        "results_root": str(results_root),
        "written_count": len(written),
        "source_counts": source_counts,
        "direct_table": str(args.direct_table),
        "default_table": str(args.default_table),
        "legacy_roots": [str(path) for path in args.legacy_root],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps({"summary": summary, "written": written}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
