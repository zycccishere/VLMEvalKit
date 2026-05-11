#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_runner_module(script_dir: Path):
    path = script_dir / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_mod", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(script_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drop eval artifacts and rerun eval for infer-complete by_setting tasks.")
    parser.add_argument("--matrix-config", type=Path, default=script_dir / "configs" / "matrix.yaml")
    parser.add_argument("--model-config", type=Path, default=script_dir / "configs" / "models.yaml")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--refresh-summary-json", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--policies", type=str, default="")
    parser.add_argument("--modes", type=str, default="")
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--gpu-ids", type=str, default="")
    parser.add_argument("--drop-answer-format", action="store_true")
    parser.add_argument("--rerun-even-if-acc", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--outer-workers", type=int, default=4)
    return parser.parse_args()


def build_live_summary(script_dir: Path, runner, out_json: Path | None) -> Path:
    target = out_json
    if target is None:
        tmp = tempfile.NamedTemporaryFile(prefix="bysetting_eval_refresh_", suffix=".json", delete=False)
        tmp.close()
        target = Path(tmp.name)
    runs_root = runner.results_root
    env = dict(os.environ)
    repo_root = str(runner.repo_root)
    existing_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}:{existing_py}" if existing_py else repo_root
    cmd = [
        sys.executable,
        str(script_dir / "scan_bysetting_results.py"),
        "--runs-root",
        str(runs_root),
        "--out-json",
        str(target),
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to refresh by_setting summary:\n{proc.stdout}")
    print(f"[SUMMARY][REFRESHED] {target}", flush=True)
    return target


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
    runner_args.transforms = ""
    runner_args.task_manifest = None
    runner_args.resume_infer = None
    runner_args.plan_only = True

    runner = mod.BenchmarkRunner(script_dir, runner_args)

    selected = []
    grouped = defaultdict(int)
    skipped_acc = []
    first_gpu = runner.node_gpu_ids[:1] if runner.node_gpu_ids else ["0"]

    summary_path = args.summary_json
    if args.refresh_summary_json is not None:
        summary_path = build_live_summary(script_dir, runner, args.refresh_summary_json)

    if summary_path is not None:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        wanted = {
            (
                record["model_key"],
                record["policy"],
                record["setting"],
                record["dataset"],
            )
            for record in payload.get("records", [])
        }
        for task in runner.tasks:
            key = (task.model_key, task.policy_key, task.mode, task.dataset)
            if key not in wanted:
                continue
            model = runner.models[task.model_key]
            runner.ensure_profile_ready(model)
            env = runner.build_env(model, task, first_gpu)
            expected = runner.get_expected_count(model, env, task.dataset)
            if expected < 0:
                continue
            if not runner.infer_complete(task, model, expected):
                continue
            if not args.rerun_even_if_acc and runner.acc_complete(task, model):
                skipped_acc.append(task.tag)
                continue
            selected.append((task, model, env, expected))
            grouped[(task.policy_key, task.mode, task.model_key)] += 1
    else:
        for task in runner.tasks:
            model = runner.models[task.model_key]
            runner.ensure_profile_ready(model)
            env = runner.build_env(model, task, first_gpu)
            expected = runner.get_expected_count(model, env, task.dataset)
            if expected < 0:
                continue
            if not runner.infer_complete(task, model, expected):
                continue
            if not args.rerun_even_if_acc and runner.acc_complete(task, model):
                skipped_acc.append(task.tag)
                continue
            selected.append((task, model, env, expected))
            grouped[(task.policy_key, task.mode, task.model_key)] += 1

    if args.limit > 0:
        selected = selected[: args.limit]

    summary = {
        "selected_tasks": len(selected),
        "skipped_existing_acc": len(skipped_acc),
        "outer_workers": args.outer_workers,
        "groups": [
            {
                "policy": policy,
                "mode": mode,
                "model": model,
                "count": count,
            }
            for (policy, mode, model), count in sorted(grouped.items())
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if args.dry_run:
        return 0

    failures = []

    def run_one(item):
        idx, (task, model, env, expected) = item
        print(f"[RERUN][{idx}/{len(selected)}] {task.tag} expected={expected}", flush=True)
        for path in runner.acc_marker_paths(task, model):
            path.unlink(missing_ok=True)
        runner.cleanup_eval_artifacts(task, model)
        if args.drop_answer_format:
            model_dir = runner.model_output_root(task, model)
            for suffix in ("_answer_format_report.json", "_answer_format_failures.jsonl"):
                path = model_dir / f"{model.registry_name}_{task.dataset}{suffix}"
                path.unlink(missing_ok=True)
        rc = runner.run_eval(task, model, env)
        return {"task": task.tag, "rc": rc}

    indexed = list(enumerate(selected, 1))
    max_workers = max(1, min(args.outer_workers, len(indexed)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_one, item) for item in indexed]
        for fut in as_completed(futures):
            result = fut.result()
            if result["rc"] != 0:
                failures.append(result)

    if failures:
        print(json.dumps({"failures": failures}, ensure_ascii=False, indent=2), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
