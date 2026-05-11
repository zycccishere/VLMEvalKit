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
        description="Build a canonical gpu_job_runner manifest for Qwen32B faithful attribution jobs."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--indices-json", required=True, help="Local path to a JSON file with an `indices` field.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--allowed-gpu-ids", default="0,1,2,3")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--score-kind", default="both", choices=["logprob", "margin", "both"])
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--runner-root", default=DEFAULT_RUNNER_ROOT)
    parser.add_argument("--probe-root", default=DEFAULT_PROBE_ROOT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--output", required=True, help="Local output manifest path.")
    return parser


def load_indices(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    indices = payload.get("indices")
    if not isinstance(indices, list) or not indices or not all(isinstance(x, int) for x in indices):
        raise SystemExit(f"Invalid indices payload in {path}")
    return indices


def build_job(
    *,
    args: argparse.Namespace,
    indices: list[int],
    corruption_family: str,
    rope_align: bool,
) -> dict:
    condition = "rope_align" if rope_align else "standard"
    sample_count = len(indices)
    name = f"{args.dataset.lower()}_{corruption_family}_{condition}_{sample_count}"
    output_dir = f"{args.probe_root}/{args.run_name}/{corruption_family}/{condition}"
    command = [
        args.python_bin,
        "scripts/qwen25vl_faithful_attribution_probe.py",
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
        "--corruption-family",
        corruption_family,
        "--device",
        "auto",
        "--output-dir",
        output_dir,
    ]
    if rope_align:
        command.append("--rope-align")
    return {
        "name": name,
        "gpus_required": 2,
        "command": command,
        "metadata": {
            "dataset": args.dataset,
            "condition": condition,
            "corruption_family": corruption_family,
            "sample_count": sample_count,
            "max_new_tokens": args.max_new_tokens,
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    indices = load_indices(Path(args.indices_json))
    manifest = {
        "name": args.run_name,
        "run_root": f"{args.runner_root}/{args.run_name}",
        "allowed_gpu_ids": [x for x in args.allowed_gpu_ids.replace(",", " ").split() if x],
        "poll_interval_sec": 15,
        "status_interval_sec": 60,
        "max_jobs_per_gpu": 1,
        "jobs": [
            build_job(args=args, indices=indices, corruption_family="blank", rope_align=False),
            build_job(args=args, indices=indices, corruption_family="blank", rope_align=True),
            build_job(args=args, indices=indices, corruption_family="dataset_swap", rope_align=False),
            build_job(args=args, indices=indices, corruption_family="dataset_swap", rope_align=True),
        ],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
