#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


CANONICAL_RUNNER_PATH = Path("/path/to/LLaVA/scripts/gpu_job_runner.py")


def resolve_runner_path(raw: Path) -> Path:
    resolved = raw.expanduser()
    if resolved != CANONICAL_RUNNER_PATH:
        raise SystemExit(
            "launch_benchmark_queue.py now only supports the canonical runner path: "
            f"{CANONICAL_RUNNER_PATH}"
        )
    return resolved


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise SystemExit("PyYAML is required for YAML queue specs.")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit(f"Queue spec must be a mapping: {path}")
    return data


def dump_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_manifest(spec: dict[str, Any], spec_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    repo_root = str(spec.get("repo_root") or spec_path.resolve().parents[1])
    jobs_raw = spec.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise SystemExit("Queue spec must contain a non-empty jobs list.")

    runner_cfg = dict(spec.get("runner", {}))
    manifest: dict[str, Any] = {
        "name": str(spec.get("name") or spec_path.stem),
        "run_root": str(args.run_root or runner_cfg.get("run_root") or (Path(repo_root) / "runs" / "gpu_job_runner" / (spec.get("name") or spec_path.stem))),
        "allowed_gpu_ids": [part for part in str(args.gpu_ids or runner_cfg.get("gpu_ids") or "0,1,2,3,4,5,6,7").replace(",", " ").split() if part],
        "poll_interval_sec": int(args.poll_interval_sec or runner_cfg.get("poll_interval_sec", 15)),
        "status_interval_sec": int(args.status_interval_sec or runner_cfg.get("status_interval_sec", 60)),
        "min_free_memory_mb": int(runner_cfg.get("min_free_memory_mb", 1000)),
        "jobs": [],
    }

    default_env = {str(key): str(value) for key, value in spec.get("env", {}).items()}
    default_matrix = str(spec.get("matrix_config") or "scripts/configs/matrix.yaml")
    default_models = str(spec.get("model_config") or "scripts/configs/models.yaml")

    for raw in jobs_raw:
        if not isinstance(raw, dict):
            raise SystemExit("Each queue job must be a mapping.")
        cmd = [
            "bash",
            "scripts/run_benchmark.sh",
            "--matrix-config",
            str(raw.get("matrix_config", default_matrix)),
            "--model-config",
            str(raw.get("model_config", default_models)),
            "--nodes",
            "1",
            "--node-rank",
            "0",
            "--gpu-ids",
            "{gpu_ids_csv}",
        ]
        for key in ("models", "policies", "modes", "datasets"):
            value = str(raw.get(key, "")).strip()
            if value:
                cmd.extend([f"--{key}", value])
        if bool(raw.get("resume_infer", False)):
            cmd.append("--resume-infer")
        else:
            cmd.append("--no-resume-infer")
        extra_args = raw.get("extra_args", [])
        if extra_args:
            if not isinstance(extra_args, list) or not all(isinstance(item, str) for item in extra_args):
                raise SystemExit(f"Job {raw.get('name', '<unnamed>')} extra_args must be a list[str].")
            cmd.extend(extra_args)

        job_env = dict(default_env)
        job_env.update({str(key): str(value) for key, value in raw.get("env", {}).items()})

        manifest["jobs"].append(
            {
                "name": str(raw["name"]),
                "cwd": repo_root,
                "gpus_required": int(raw.get("gpus_required", 1)),
                "min_free_memory_mb": int(raw.get("min_free_memory_mb", manifest["min_free_memory_mb"])),
                "retries": int(raw.get("retries", 0)),
                "env": job_env,
                "metadata": {key: raw.get(key) for key in ("models", "policies", "modes", "datasets")},
                "command": cmd,
            }
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and optionally launch a GPU-runner queue for VLMEvalKit benchmark jobs.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--runner-path", type=Path, default=CANONICAL_RUNNER_PATH)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--gpu-ids", type=str, default="")
    parser.add_argument("--poll-interval-sec", type=int, default=None)
    parser.add_argument("--status-interval-sec", type=int, default=None)
    parser.add_argument("--launch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.runner_path = resolve_runner_path(args.runner_path)
    spec = load_config(args.spec.resolve())
    manifest = build_manifest(spec, args.spec.resolve(), args)
    manifest_out = args.manifest_out or (Path(manifest["run_root"]) / "manifest.json")
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    dump_json(manifest_out, manifest)
    print(f"manifest={manifest_out}", flush=True)
    print(f"run_root={manifest['run_root']}", flush=True)
    if not args.launch:
        return 0

    cmd = [sys.executable, str(args.runner_path), "--manifest", str(manifest_out)]
    if args.poll_interval_sec is not None:
        cmd.extend(["--poll-interval-sec", str(args.poll_interval_sec)])
    if args.status_interval_sec is not None:
        cmd.extend(["--status-interval-sec", str(args.status_interval_sec)])
    print("launch=" + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
