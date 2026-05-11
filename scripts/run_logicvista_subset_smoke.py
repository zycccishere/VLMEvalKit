#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small LogicVista replay smoke with a sliced dataset."
    )
    parser.add_argument("--model", default="Qwen3.5-4B-Replay")
    parser.add_argument("--dataset", default="LogicVista")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--policy", default="identity", choices=["identity", "directly_answer"])
    parser.add_argument("--mode", default="image_text")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--judge", default="gpt-4o-mini")
    parser.add_argument("--api-nproc", type=int, default=4)
    parser.add_argument("--eval-nproc", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--openai-api-key", default="")
    parser.add_argument("--openai-api-base", default="")
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def set_runtime_env(args: argparse.Namespace) -> None:
    os.environ["REPLAY_MODE"] = args.mode
    os.environ["REPLAY_PROMPT_TEMPLATE_NAME"] = args.policy
    os.environ["REPLAY_TIMES"] = "1"
    os.environ["REPLAY_IMAGE_COPY_MODE"] = "reuse_path"
    os.environ["REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT"] = "1"
    os.environ.setdefault("REPLAY_LIMIT_MM_PER_PROMPT", "2")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
        os.environ["OPENAI_API_KEY_JUDGE"] = args.openai_api_key
    if args.openai_api_base:
        os.environ["OPENAI_API_BASE"] = args.openai_api_base
        os.environ["OPENAI_API_BASE_JUDGE"] = args.openai_api_base


def cleanup_outputs(pred_root: Path, model_name: str, dataset_name: str) -> None:
    for path in pred_root.glob(f"{model_name}_{dataset_name}*"):
        if path.is_file():
            path.unlink()


def main() -> int:
    args = build_parser().parse_args()
    set_runtime_env(args)

    from vlmeval.dataset import build_dataset
    from vlmeval.inference import infer_data_job

    dataset = build_dataset(args.dataset)
    if dataset is None:
        raise RuntimeError(f"Failed to build dataset: {args.dataset}")

    dataset.data = dataset.data.iloc[: args.limit].copy().reset_index(drop=True)
    pred_root = Path(args.work_dir).expanduser().resolve() / args.model
    pred_root.mkdir(parents=True, exist_ok=True)

    if args.force_clean:
        cleanup_outputs(pred_root, args.model, dataset.dataset_name)

    infer_data_job(
        model=args.model,
        work_dir=str(pred_root),
        model_name=args.model,
        dataset=dataset,
        verbose=args.verbose,
        api_nproc=args.api_nproc,
        batch_size=args.batch_size,
    )

    result_file = pred_root / f"{args.model}_{dataset.dataset_name}.xlsx"
    score = dataset.evaluate(
        str(result_file),
        model=args.judge,
        nproc=args.eval_nproc,
    )

    summary = {
        "model": args.model,
        "dataset": dataset.dataset_name,
        "limit": int(args.limit),
        "policy": args.policy,
        "mode": args.mode,
        "result_file": str(result_file),
        "score_type": type(score).__name__,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
