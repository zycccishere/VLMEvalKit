#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MODEL_PATH = "/models/Qwen2.5-VL-32B-Instruct"
DEFAULT_PYTHON = "/opt/miniconda3/envs/vlmevalkit/bin/python"
DEFAULT_REMOTE_ROOT = "/path/to/vlmevalkit"
DEFAULT_RUNNER_ROOT = "/path/to/LLaVA/runs/gpu_job_runner"
DEFAULT_PROBE_ROOT = "/path/to/vlmevalkit/runs/probes"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a canonical gpu_job_runner manifest for Qwen32B value-zeroing jobs."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--indices-json", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--allowed-gpu-ids", default="4,5,6,7")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--score-kind", default="both", choices=["logprob", "margin", "both"])
    parser.add_argument("--zero-layers", default="all")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--runner-root", default=DEFAULT_RUNNER_ROOT)
    parser.add_argument("--probe-root", default=DEFAULT_PROBE_ROOT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--output", required=True)
    return parser


def load_indices(path: Path, max_samples: int) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    indices = payload.get("indices")
    if not isinstance(indices, list) or not indices or not all(isinstance(x, int) for x in indices):
        raise SystemExit(f"Invalid indices payload in {path}")
    if max_samples > 0:
        indices = indices[:max_samples]
    return indices


def build_job(
    *,
    args: argparse.Namespace,
    indices: list[int],
    rope_align: bool,
) -> dict:
    condition = "rope_align" if rope_align else "standard"
    sample_count = len(indices)
    output_dir = f"{args.probe_root}/{args.run_name}/{condition}"
    command = [
        args.python_bin,
        "scripts/qwen25vl_value_zeroing_probe.py",
        "--model-path",
        args.model_path,
        "--dataset",
        args.dataset,
        "--indices",
        *[str(x) for x in indices],
        "--mode",
        "image_text_image",
        "--policy",
        "identity",
        "--template-on-last-replay-text",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--score-kind",
        args.score_kind,
        "--zero-layers",
        args.zero_layers,
        "--device",
        "auto",
        "--output-dir",
        output_dir,
    ]
    if rope_align:
        command.append("--rope-align")
    return {
        "name": f"{args.dataset.lower()}_{condition}_{sample_count}_value_zeroing",
        "cwd": args.remote_root,
        "gpus_required": 2,
        "min_free_memory_mb": 60000,
        "retries": 0,
        "env": {
            "PYTHONPATH": args.remote_root,
            "VLMEVAL_VLM_MINIMAL_IMPORT": "1",
            "VLMEVAL_API_MINIMAL_IMPORT": "1",
            "VLMEVAL_LAZY_INIT": "1",
        },
        "metadata": {
            "dataset": args.dataset,
            "condition": condition,
            "sample_count": sample_count,
            "max_new_tokens": args.max_new_tokens,
            "score_kind": args.score_kind,
            "zero_layers": args.zero_layers,
        },
        "command": command,
    }


def main() -> int:
    args = build_parser().parse_args()
    indices = load_indices(Path(args.indices_json), args.max_samples)
    manifest = {
        "name": args.run_name,
        "run_root": f"{args.runner_root}/{args.run_name}",
        "allowed_gpu_ids": [x for x in args.allowed_gpu_ids.replace(",", " ").split() if x],
        "poll_interval_sec": 15,
        "status_interval_sec": 60,
        "max_jobs_per_gpu": 1,
        "min_free_memory_mb": 1000,
        "jobs": [
            build_job(args=args, indices=indices, rope_align=False),
            build_job(args=args, indices=indices, rope_align=True),
        ],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
