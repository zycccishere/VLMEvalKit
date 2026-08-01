#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


COUNTQA_EXPECTED = {
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
    checked_countqa_records = 0
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
            if dataset != "CountQA":
                continue
            checked_countqa_records += 1
            for key, expected in COUNTQA_EXPECTED.items():
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
    if checked_countqa_records == 0:
        errors.append("no CountQA final_model_input records found")

    required_tasks = list(dict.fromkeys(args.required_task))
    for task_key in required_tasks:
        if observed[task_key] == 0:
            errors.append(f"required task has no final input records: {task_key}")
        elif observed[task_key] != args.expect_records_per_task:
            errors.append(
                f"required task {task_key} has {observed[task_key]} final input records, "
                f"expected {args.expect_records_per_task}"
            )

    payload = {
        "root": str(args.root),
        "checked_final_records": checked_records,
        "checked_countqa_final_records": checked_countqa_records,
        "observed_tasks": dict(sorted(observed.items())),
        "required_tasks": required_tasks,
        "expected_records_per_task": args.expect_records_per_task,
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
