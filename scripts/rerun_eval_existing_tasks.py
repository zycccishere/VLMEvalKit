#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - optional dependency in some envs
    dotenv_values = None


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML: {path}")
    return data


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_repo_dotenv(repo_root: Path) -> None:
    """Load repo-root .env into the current process so rerun eval matches run_benchmark.sh."""
    env_path = repo_root / ".env"
    if not env_path.exists() or dotenv_values is None:
        return
    values = dotenv_values(env_path)
    for key, value in values.items():
        if key and value:
            os.environ[key] = str(value)


def split_names(raw: str) -> list[str]:
    if not raw:
        return []
    return [part for part in raw.replace(",", " ").split() if part]


@dataclass(frozen=True)
class EnvProfile:
    python: str
    pythonpath: tuple[str, ...]
    base_env: dict[str, str]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    registry_name: str
    env_profile: str


@dataclass(frozen=True)
class EvalTask:
    dataset: str
    model_key: str
    model_name: str
    task_root: Path
    pred_file: Path
    policy: str
    mode: str
    transform: str

    @property
    def tag(self) -> str:
        return f"{self.model_key}__{self.policy}__{self.mode}__{self.transform}__{self.dataset}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun eval for existing infer outputs.")
    parser.add_argument("--matrix-config", required=True)
    parser.add_argument("--model-config", default="scripts/configs/models.yaml")
    parser.add_argument("--datasets", default="DynaMath,VisualPuzzles")
    parser.add_argument("--policies", default="")
    parser.add_argument("--modes", default="")
    parser.add_argument("--transforms", default="")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--judge", default=None)
    parser.add_argument("--nproc", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_env(repo_root: Path, matrix: dict[str, Any], profile: EnvProfile) -> dict[str, str]:
    env = dict(os.environ)
    env.update(profile.base_env)
    pythonpath = [str(Path(entry.format(repo_root=str(repo_root)))) for entry in profile.pythonpath if entry]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(pythonpath + ([existing] if existing else []))
    eval_cfg = matrix.get("evaluation", {}) or {}
    api_key = str(
        os.environ.get("OPENAI_API_KEY_JUDGE")
        or os.environ.get("OPENAI_API_KEY")
        or eval_cfg.get("openai_api_key", "")
    ).strip()
    api_base = str(
        os.environ.get("OPENAI_API_BASE_JUDGE")
        or os.environ.get("OPENAI_API_BASE")
        or eval_cfg.get("openai_api_base", "")
    ).strip()
    if api_key:
        env["OPENAI_API_KEY"] = api_key
        env["OPENAI_API_KEY_JUDGE"] = api_key
    if api_base:
        env["OPENAI_API_BASE"] = api_base
        env["OPENAI_API_BASE_JUDGE"] = api_base
    return env


def iter_eval_artifacts(model_dir: Path, model_name: str, dataset: str) -> list[Path]:
    patterns = [
        f"{model_name}_{dataset}_gpt*",
        f"{model_name}_{dataset}_chatgpt*",
        f"{model_name}_{dataset}_exact_matching*",
        f"{model_name}_{dataset}_acc.csv",
        f"{model_name}_{dataset}_score.csv",
        f"{model_name}_{dataset}_score.json",
    ]
    out: list[Path] = []
    for path in model_dir.iterdir():
        if not path.is_file():
            continue
        if any(fnmatch.fnmatch(path.name, pat) for pat in patterns):
            out.append(path)
    return sorted(out)


def discover_tasks(
    repo_root: Path,
    matrix: dict[str, Any],
    models_cfg: dict[str, Any],
    datasets: set[str],
    policies: set[str] | None = None,
    modes: set[str] | None = None,
    transforms: set[str] | None = None,
) -> list[EvalTask]:
    results_root_raw = str(matrix["results_root"])
    results_root = Path(results_root_raw) if Path(results_root_raw).is_absolute() else repo_root / results_root_raw
    models = []
    for model_key in matrix.get("models", []):
        spec = models_cfg["models"][model_key]
        models.append(
            ModelSpec(
                key=model_key,
                registry_name=spec["registry_name"],
                env_profile=spec["env_profile"],
            )
        )
    found: list[EvalTask] = []
    for model in models:
        for dataset in sorted(datasets):
            pattern = f"**/{model.registry_name}/{model.registry_name}_{dataset}.xlsx"
            for pred_file in sorted(results_root.glob(pattern)):
                task_root = pred_file.parent.parent
                rel = task_root.relative_to(results_root)
                parts = rel.parts
                if len(parts) == 3:
                    policy, mode, model_key = parts
                    transform = "baseline"
                elif len(parts) == 4:
                    policy, mode, transform, model_key = parts
                else:
                    raise ValueError(f"Unexpected task root layout: {task_root}")
                if policies and policy not in policies:
                    continue
                if modes and mode not in modes:
                    continue
                if transforms and transform not in transforms:
                    continue
                found.append(
                    EvalTask(
                        dataset=dataset,
                        model_key=model_key,
                        model_name=model.registry_name,
                        task_root=task_root,
                        pred_file=pred_file,
                        policy=policy,
                        mode=mode,
                        transform=transform,
                    )
                )
    found.sort(key=lambda t: (t.dataset != "DynaMath", t.mode, t.transform, t.dataset, t.model_key))
    return found


def run_one(task: EvalTask, repo_root: Path, python_bin: str, env: dict[str, str], nproc: int, judge: str | None) -> dict[str, Any]:
    model_dir = task.pred_file.parent
    stale = iter_eval_artifacts(model_dir, task.model_name, task.dataset)
    removed = []
    for path in stale:
        path.unlink()
        removed.append(path.name)

    log_dir = task.task_root / "_logs" / "eval_rerun"
    ensure_dir(log_dir)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    log_path = log_dir / f"{task.model_name}_{task.dataset}_{ts}.log"
    cmd = [
        python_bin,
        "run.py",
        "--data",
        task.dataset,
        "--model",
        task.model_name,
        "--work-dir",
        str(task.task_root),
        "--mode",
        "eval",
        "--nproc",
        str(nproc),
        "--verbose",
    ]
    if judge:
        cmd.extend(["--judge", judge])

    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("[CMD] " + " ".join(cmd) + "\n")
        fh.write("[REMOVED] " + json.dumps(removed, ensure_ascii=False) + "\n")
        fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

    remaining = [p.name for p in iter_eval_artifacts(model_dir, task.model_name, task.dataset)]
    return {
        "task": task,
        "rc": proc.returncode,
        "log_path": str(log_path),
        "removed": removed,
        "remaining_artifacts": remaining,
    }


def main() -> int:
    args = parse_args()
    matrix_path = Path(args.matrix_config)
    model_cfg_path = Path(args.model_config)
    matrix = load_yaml(matrix_path)
    models_cfg = load_yaml(model_cfg_path)
    repo_root = Path(matrix["repo_root"])
    load_repo_dotenv(repo_root)
    datasets = set(split_names(args.datasets))
    policy_filters = set(split_names(args.policies))
    mode_filters = set(split_names(args.modes))
    transform_filters = set(split_names(args.transforms))
    tasks = discover_tasks(
        repo_root,
        matrix,
        models_cfg,
        datasets,
        policies=policy_filters or None,
        modes=mode_filters or None,
        transforms=transform_filters or None,
    )
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        print("[INFO] no matching tasks found", flush=True)
        return 0

    first_model_key = tasks[0].model_key
    model_spec = models_cfg["models"][first_model_key]
    profile_raw = models_cfg["env_profiles"][model_spec["env_profile"]]
    profile = EnvProfile(
        python=profile_raw["python"],
        pythonpath=tuple(profile_raw.get("pythonpath", [])),
        base_env=dict(profile_raw.get("base_env", {})),
    )
    env = build_env(repo_root, matrix, profile)
    judge = args.judge if args.judge is not None else str((matrix.get("evaluation", {}) or {}).get("judge", "gpt-4o-mini"))
    nproc = args.nproc if args.nproc is not None else int((matrix.get("evaluation", {}) or {}).get("nproc", 8))

    print(
        json.dumps(
            {
                "repo_root": str(repo_root),
                "task_count": len(tasks),
                "datasets": sorted(datasets),
                "max_workers": args.max_workers,
                "judge": judge,
                "nproc": nproc,
                "sample_tags": [t.tag for t in tasks[:8]],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0

    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        future_map = {ex.submit(run_one, task, repo_root, profile.python, env, nproc, judge): task for task in tasks}
        for future in as_completed(future_map):
            result = future.result()
            task = result["task"]
            print(
                json.dumps(
                    {
                        "tag": task.tag,
                        "rc": result["rc"],
                        "removed": result["removed"],
                        "remaining_artifacts": result["remaining_artifacts"],
                        "log_path": result["log_path"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if result["rc"] != 0:
                failures.append(result)

    if failures:
        print(f"[FAIL] {len(failures)} task(s) failed", flush=True)
        for item in failures:
            print(f"  - {item['task'].tag}: {item['log_path']}", flush=True)
        return 1
    print("[DONE] all rerun eval tasks finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
