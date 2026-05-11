#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from rope_probe_jobs import (
    REMOTE_VLMEVAL_ROOT,
    build_command,
    build_job_env,
    build_model_specs,
    estimated_minutes_for_dataset,
    indices_for_dataset,
    split_csv,
)


CANONICAL_RUNNER_PATH = Path("/path/to/LLaVA/scripts/gpu_job_runner.py")


def dump_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and optionally launch a gpu_job_runner manifest for the rope-only image_text_image probe sweep."
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--models", default="qwen25vl_32b,llava_aligned_iti,llava_aligned_it")
    parser.add_argument("--datasets", default="AI2D_TEST,LogicVista,MathVision,DynaMath")
    parser.add_argument("--conditions", default="standard,rope_align")
    parser.add_argument(
        "--output-root",
        default=str(REMOTE_VLMEVAL_ROOT / "runs" / "probes" / "rope_align_image_text_image_20260330"),
    )
    parser.add_argument(
        "--run-root",
        default=str(REMOTE_VLMEVAL_ROOT / "runs" / "gpu_job_runner" / "rope_align_image_text_image_probe_queue_20260401"),
    )
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--runner-path", default=str(CANONICAL_RUNNER_PATH))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--attn-layers", default="last")
    parser.add_argument("--head-reduction", default="mean", choices=["per_head", "mean"])
    parser.add_argument("--poll-interval-sec", type=int, default=15)
    parser.add_argument("--status-interval-sec", type=int, default=60)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--launch", action="store_true")
    return parser


def build_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    requested_models = split_csv(args.models)
    requested_datasets = split_csv(args.datasets)
    requested_conditions = split_csv(args.conditions)
    allowed_gpu_ids = split_csv(args.gpu_ids)
    specs = build_model_specs()

    for name in requested_models:
        if name not in specs:
            raise SystemExit(f"Unknown model key: {name}")
    for condition in requested_conditions:
        if condition not in {"standard", "rope_align"}:
            raise SystemExit(f"Unsupported condition: {condition}")

    output_root = Path(args.output_root)
    jobs: list[dict[str, Any]] = []
    estimated_wall_minutes = 0
    summary_rows = []
    for dataset in requested_datasets:
        indices = indices_for_dataset(dataset, max_samples=args.max_samples)
        estimated_wall_minutes += estimated_minutes_for_dataset(dataset, len(indices))
        for model_key in requested_models:
            spec = specs[model_key]
            for condition in requested_conditions:
                cmd = build_command(
                    spec=spec,
                    dataset=dataset,
                    condition=condition,
                    indices=indices,
                    output_root=output_root,
                    attn_layers=args.attn_layers,
                    head_reduction=args.head_reduction,
                )
                jobs.append(
                    {
                        "name": f"{dataset}__{model_key}__{condition}",
                        "cwd": str(spec.repo_root),
                        "gpus_required": spec.gpus_per_job,
                        "min_free_memory_mb": spec.min_free_memory_mb,
                        "retries": 0,
                        "env": build_job_env(spec),
                        "metadata": {
                            "dataset": dataset,
                            "model": model_key,
                            "condition": condition,
                            "sample_count": len(indices),
                            "attn_layers": args.attn_layers,
                            "head_reduction": args.head_reduction,
                            "output_dir": str(output_root / dataset / spec.key / condition),
                        },
                        "command": cmd,
                    }
                )
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "model": model_key,
                        "condition": condition,
                        "gpus_required": spec.gpus_per_job,
                        "sample_count": len(indices),
                        "cwd": str(spec.repo_root),
                        "cmd": " ".join(cmd),
                    }
                )

    manifest = {
        "name": Path(args.run_root).name,
        "run_root": str(Path(args.run_root)),
        "allowed_gpu_ids": allowed_gpu_ids,
        "poll_interval_sec": args.poll_interval_sec,
        "status_interval_sec": args.status_interval_sec,
        "min_free_memory_mb": 1000,
        "jobs": jobs,
    }
    summary = {
        "run_root": str(Path(args.run_root)),
        "output_root": str(output_root),
        "runner_path": str(Path(args.runner_path)),
        "datasets": requested_datasets,
        "models": requested_models,
        "conditions": requested_conditions,
        "allowed_gpu_ids": allowed_gpu_ids,
        "estimated_wall_minutes": estimated_wall_minutes,
        "job_count": len(jobs),
        "jobs": summary_rows,
    }
    return manifest, summary


def main() -> int:
    args = build_parser().parse_args()
    manifest, summary = build_manifest(args)

    run_root = Path(manifest["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_out = Path(args.manifest_out) if args.manifest_out else run_root / "manifest.json"
    dump_json(manifest_out, manifest)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"manifest={manifest_out}", flush=True)

    if args.plan_only or not args.launch:
        return 0

    launch_cmd = [sys.executable, str(Path(args.runner_path)), "--manifest", str(manifest_out)]
    if args.poll_interval_sec:
        launch_cmd.extend(["--poll-interval-sec", str(args.poll_interval_sec)])
    if args.status_interval_sec:
        launch_cmd.extend(["--status-interval-sec", str(args.status_interval_sec)])
    print("launch=" + " ".join(launch_cmd), flush=True)
    return subprocess.run(launch_cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
