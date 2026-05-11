#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from rope_probe_jobs import (
    REMOTE_VLMEVAL_ROOT,
    build_command,
    build_job_env,
    build_model_specs,
    estimated_minutes_for_dataset,
    indices_for_dataset,
    split_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the 2026-03-30 rope-only image_text_image probe sweep on 8 GPUs."
    )
    parser.add_argument(
        "--gpu-ids",
        default="0,1,2,3,4,5,6,7",
        help="Exactly eight GPU ids on the current node.",
    )
    parser.add_argument(
        "--models",
        default="qwen25vl_32b,llava_aligned_iti,llava_aligned_it",
    )
    parser.add_argument(
        "--datasets",
        default="AI2D_TEST,LogicVista,MathVision,DynaMath",
    )
    parser.add_argument(
        "--conditions",
        default="standard,rope_align",
    )
    parser.add_argument(
        "--output-root",
        default=str(REMOTE_VLMEVAL_ROOT / "runs" / "probes" / "rope_align_image_text_image_20260330"),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional smoke cap applied after loading each fixed manifest.",
    )
    parser.add_argument(
        "--attn-layers",
        default="last",
        help="Attention layers to trace in each probe backend.",
    )
    parser.add_argument(
        "--head-reduction",
        default="mean",
        choices=["per_head", "mean"],
    )
    parser.add_argument("--plan-only", action="store_true")
    return parser


def slot_assignment(gpu_ids: list[str]) -> dict[tuple[str, str], list[str]]:
    if len(gpu_ids) != 8:
        raise SystemExit(f"--gpu-ids must resolve to exactly 8 entries, got {gpu_ids}")
    return {
        ("qwen25vl_32b", "standard"): gpu_ids[0:2],
        ("qwen25vl_32b", "rope_align"): gpu_ids[2:4],
        ("llava_aligned_iti", "standard"): gpu_ids[4:5],
        ("llava_aligned_iti", "rope_align"): gpu_ids[5:6],
        ("llava_aligned_it", "standard"): gpu_ids[6:7],
        ("llava_aligned_it", "rope_align"): gpu_ids[7:8],
    }


def main() -> int:
    args = build_parser().parse_args()
    requested_models = split_csv(args.models)
    requested_datasets = split_csv(args.datasets)
    requested_conditions = split_csv(args.conditions)
    gpu_ids = split_csv(args.gpu_ids)
    specs = build_model_specs()
    slots = slot_assignment(gpu_ids)
    output_root = Path(args.output_root)

    for name in requested_models:
        if name not in specs:
            raise SystemExit(f"Unknown model key: {name}")
    for condition in requested_conditions:
        if condition not in {"standard", "rope_align"}:
            raise SystemExit(f"Unsupported condition: {condition}")

    total_estimate = 0
    plan_rows = []
    for dataset in requested_datasets:
        indices = indices_for_dataset(dataset, max_samples=args.max_samples)
        total_estimate += estimated_minutes_for_dataset(dataset, len(indices))
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
                plan_rows.append(
                    {
                        "dataset": dataset,
                        "model": model_key,
                        "condition": condition,
                        "gpu_ids": slots[(model_key, condition)],
                        "cwd": str(spec.repo_root),
                        "cmd": cmd,
                    }
                )

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "datasets": requested_datasets,
                "models": requested_models,
                "conditions": requested_conditions,
                "estimated_wall_minutes": total_estimate,
                "job_count": len(plan_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for row in plan_rows:
        printable = " ".join(shlex.quote(part) for part in row["cmd"])
        print(
            json.dumps(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "condition": row["condition"],
                    "gpu_ids": row["gpu_ids"],
                    "cwd": row["cwd"],
                    "cmd": printable,
                },
                ensure_ascii=False,
            )
        )

    if args.plan_only:
        return 0

    failures = []
    for dataset in requested_datasets:
        procs = []
        for model_key in requested_models:
            spec = specs[model_key]
            indices = indices_for_dataset(dataset, max_samples=args.max_samples)
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
                env = dict(os.environ)
                env.update(build_job_env(spec))
                env["CUDA_VISIBLE_DEVICES"] = ",".join(slots[(model_key, condition)])
                log_path = output_root / "_logs" / dataset / f"{model_key}__{condition}.log"
                print(
                    f"[launch] dataset={dataset} model={model_key} condition={condition} "
                    f"gpus={','.join(slots[(model_key, condition)])}",
                    flush=True,
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                fh = log_path.open("w", encoding="utf-8")
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(spec.repo_root),
                    env=env,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                procs.append((dataset, model_key, condition, proc, fh, log_path))

        for dataset_name, model_key, condition, proc, fh, log_path in procs:
            rc = proc.wait()
            fh.close()
            print(
                f"[done] dataset={dataset_name} model={model_key} condition={condition} rc={rc} log={log_path}",
                flush=True,
            )
            if rc != 0:
                failures.append((dataset_name, model_key, condition, rc, str(log_path)))

    if failures:
        for dataset_name, model_key, condition, rc, log_path in failures:
            print(
                f"[fail] dataset={dataset_name} model={model_key} condition={condition} rc={rc} log={log_path}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
