#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_runner_module(script_dir: Path):
    path = script_dir / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_mod", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(script_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the current set of by_setting tasks that are infer-complete but still need eval."
    )
    parser.add_argument("--matrix-config", type=Path, default=script_dir / "configs" / "matrix.yaml")
    parser.add_argument("--model-config", type=Path, default=script_dir / "configs" / "models.yaml")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--policies", type=str, default="")
    parser.add_argument("--modes", type=str, default="")
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--gpu-ids", type=str, default="")
    return parser.parse_args()


def build_expected_counts(datasets: list[str]) -> dict[str, int]:
    import contextlib
    import io

    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from vlmeval.dataset import build_dataset

    counts: dict[str, int] = {}
    for name in datasets:
        buf = io.StringIO()
        dataset = None
        err = None
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                dataset = build_dataset(name)
            except Exception as exc:
                err = exc
        if dataset is None or err is not None:
            counts[name] = -1
            continue
        try:
            counts[name] = int(len(dataset))
        except Exception:
            data = getattr(dataset, "data", None)
            counts[name] = int(len(data)) if data is not None else -1
    return counts


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    args = parse_args(script_dir)
    mod = load_runner_module(script_dir)

    class RunnerArgs:
        pass

    runner_args = RunnerArgs()
    runner_args.matrix_config = args.matrix_config
    runner_args.model_config = args.model_config
    runner_args.nodes = 1
    runner_args.node_rank = 0
    runner_args.gpu_ids = args.gpu_ids
    runner_args.models = args.models
    runner_args.policies = args.policies
    runner_args.modes = args.modes
    runner_args.datasets = args.datasets
    runner_args.resume_infer = None
    runner_args.plan_only = True

    runner = mod.BenchmarkRunner(script_dir, runner_args)
    expected_counts = build_expected_counts(sorted({task.dataset for task in runner.tasks}))

    records: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str], int] = defaultdict(int)
    skipped_infer = 0
    skipped_eval_complete = 0

    for task in runner.tasks:
        model = runner.models[task.model_key]
        expected = expected_counts.get(task.dataset, -1)
        if expected < 0:
            continue
        if not runner.infer_complete(task, model, expected):
            skipped_infer += 1
            continue
        if runner.eval_complete(task, model, expected):
            skipped_eval_complete += 1
            continue
        records.append(
            {
                "policy": task.policy_key,
                "setting": task.mode,
                "model_key": task.model_key,
                "model_label": model.display_name,
                "registry_name": model.registry_name,
                "dataset": task.dataset,
                "expected_count": expected,
                "task_tag": task.tag,
            }
        )
        grouped[(task.policy_key, task.mode, task.model_key)] += 1

    payload = {
        "matrix_config": str(args.matrix_config),
        "model_config": str(args.model_config),
        "selected_task_count": len(records),
        "skipped_infer_incomplete": skipped_infer,
        "skipped_eval_complete": skipped_eval_complete,
        "groups": [
            {
                "policy": policy,
                "mode": mode,
                "model_key": model_key,
                "count": count,
            }
            for (policy, mode, model_key), count in sorted(grouped.items())
        ],
        "records": records,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
