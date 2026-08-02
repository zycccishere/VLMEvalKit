#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd


GREEDY_32_EXPECTED = {
    "n": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 32,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "decoding_mode": "greedy",
    "summary_source_type": "SamplingParams",
    "summary_completeness": "selected_effective_fields",
}

CONDITION_TO_MODE = {
    "iq": "image_text",
    "iqiq": "image_text_image_text",
}

MODEL_CONTRACTS = {
    "qwen": ("qwen2.5-vl", "vllm.LLM.generate"),
    "gemma": ("gemma3", "vllm.LLM.generate"),
    "minicpm": ("minicpm-v/o-4.5", "vllm.LLM.chat"),
}


def _equal(actual, expected):
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _load_predictions(path):
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, usecols=["index", "prediction"])
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t", usecols=["index", "prediction"])
    return pd.read_csv(path, usecols=["index", "prediction"])


def _classify_normalized_bbox(value):
    number = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    separator = r"(?:\s*,\s*|\s+)"
    match = re.fullmatch(
        rf"\[\s*({number}){separator}({number}){separator}"
        rf"({number}){separator}({number})\s*\]",
        str(value).strip(),
    )
    if match is None:
        return "invalid_syntax"
    coords = [float(match.group(index)) for index in range(1, 5)]
    if not all(math.isfinite(coord) for coord in coords):
        return "invalid_syntax"
    if any(coord < 0.0 or coord > 1.0 for coord in coords):
        return "out_of_range_or_reversed"
    if coords[2] < coords[0] or coords[3] < coords[1]:
        return "out_of_range_or_reversed"
    return "valid_normalized"


def _audit_refcoco_predictions(root, required_tasks, expected_records, errors):
    audits = {}
    for task_key in required_tasks:
        dataset, model_key, condition = task_key.split(":", 2)
        if dataset != "RefCOCO":
            continue
        mode = CONDITION_TO_MODE.get(condition)
        if mode is None:
            errors.append(f"unsupported RefCOCO condition in required task: {task_key}")
            continue
        prediction_root = root / "default" / mode / "baseline" / model_key / dataset / "predictions"
        candidates = sorted(
            path for path in prediction_root.glob("*")
            if path.is_file() and path.suffix.lower() in {".xlsx", ".xls", ".csv", ".tsv"}
        )
        if len(candidates) != 1:
            errors.append(
                f"{task_key}: expected one RefCOCO prediction file, found {len(candidates)}: "
                f"{[str(path) for path in candidates]}"
            )
            continue
        path = candidates[0]
        try:
            frame = _load_predictions(path)
        except Exception as exc:
            errors.append(f"{task_key}: unable to load {path}: {type(exc).__name__}: {exc}")
            continue
        if len(frame) != expected_records:
            errors.append(
                f"{task_key}: RefCOCO predictions={len(frame)}, expected {expected_records}"
            )
        if frame["index"].astype(str).duplicated().any():
            errors.append(f"{task_key}: duplicate RefCOCO prediction indices in {path}")
        counts = Counter(_classify_normalized_bbox(value) for value in frame["prediction"])
        audits[task_key] = {
            "prediction_file": str(path),
            "records": len(frame),
            "valid_normalized": counts["valid_normalized"],
            "out_of_range_or_reversed": counts["out_of_range_or_reversed"],
            "invalid_syntax": counts["invalid_syntax"],
            "predictions": [str(value) for value in frame["prediction"].tolist()],
        }
    return audits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--required-task", action="append", required=True)
    parser.add_argument("--expect-records-per-task", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    errors = []
    observed = Counter()
    observed_indices = {}
    checked_records = 0
    checked_generation_records = 0
    for path in sorted(args.root.rglob("replay_raw.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            record = json.loads(line)
            if record.get("stage") != "final_model_input":
                continue
            identity = record.get("task_identity") or {}
            dataset = identity.get("dataset")
            model_key = identity.get("model_key")
            condition = identity.get("condition")
            task_key = f"{dataset}:{model_key}:{condition}"
            checked_records += 1
            observed[task_key] += 1
            canonical_index = identity.get("canonical_index")
            index_key = (task_key, canonical_index)
            if not canonical_index:
                errors.append(f"{path}:{line_number}:{task_key}: missing canonical_index")
            elif index_key in observed_indices:
                errors.append(
                    f"{path}:{line_number}:{task_key}: duplicate canonical_index={canonical_index!r}; "
                    f"first seen at {observed_indices[index_key]}"
                )
            else:
                observed_indices[index_key] = f"{path}:{line_number}"
            config = record.get("generation_config")
            record_prefix = f"{path}:{line_number}:{task_key}"
            model_prefix = next(
                (
                    candidate
                    for candidate in MODEL_CONTRACTS
                    if str(model_key).lower().startswith(candidate)
                ),
                None,
            )
            if model_prefix is None:
                errors.append(f"{record_prefix}: unsupported model_key={model_key!r}")
            else:
                expected_family, expected_api = MODEL_CONTRACTS[model_prefix]
                if record.get("backend") != "vllm":
                    errors.append(
                        f"{record_prefix}: backend={record.get('backend')!r}, expected 'vllm'"
                    )
                if record.get("consumer_api") != expected_api:
                    errors.append(
                        f"{record_prefix}: consumer_api={record.get('consumer_api')!r}, "
                        f"expected {expected_api!r}"
                    )
                if record.get("model_family") != expected_family:
                    errors.append(
                        f"{record_prefix}: model_family={record.get('model_family')!r}, "
                        f"expected {expected_family!r}"
                    )
            if not isinstance(config, dict):
                errors.append(f"{record_prefix}: missing generation_config")
                continue
            checked_generation_records += 1
            for key, expected in GREEDY_32_EXPECTED.items():
                if not _equal(config.get(key), expected):
                    errors.append(
                        f"{record_prefix}: generation_config.{key}={config.get(key)!r}, "
                        f"expected {expected!r}"
                    )
            if config.get("top_k") not in (0, -1):
                errors.append(
                    f"{record_prefix}: top_k={config.get('top_k')!r}, expected disabled (0 or -1)"
                )
            if config.get("best_of") not in (None, 1):
                errors.append(
                    f"{record_prefix}: best_of={config.get('best_of')!r}, expected 1 or None"
                )
            if str(model_key).startswith("minicpm") and config.get("requested_num_beams") != 1:
                errors.append(
                    f"{record_prefix}: requested_num_beams={config.get('requested_num_beams')!r}, "
                    "expected 1"
                )

    if checked_records == 0:
        errors.append("no final_model_input records found")
    if checked_generation_records == 0:
        errors.append("no generation configs found")

    required_tasks = list(dict.fromkeys(args.required_task))
    for task_key in required_tasks:
        if observed[task_key] == 0:
            errors.append(f"required task has no final input records: {task_key}")
        elif observed[task_key] != args.expect_records_per_task:
            errors.append(
                f"required task {task_key} has {observed[task_key]} final input records, "
                f"expected {args.expect_records_per_task}"
            )

    refcoco_prediction_audit = _audit_refcoco_predictions(
        args.root,
        required_tasks,
        args.expect_records_per_task,
        errors,
    )

    payload = {
        "root": str(args.root),
        "checked_final_records": checked_records,
        "checked_generation_config_records": checked_generation_records,
        "observed_tasks": dict(sorted(observed.items())),
        "required_tasks": required_tasks,
        "expected_records_per_task": args.expect_records_per_task,
        "refcoco_prediction_audit": refcoco_prediction_audit,
        "errors": errors,
        "ok": not errors,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
