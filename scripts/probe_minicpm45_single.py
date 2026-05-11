#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe one MiniCPM 4.5 replay sample.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="VisualPuzzles")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--mode", default="image_image_text")
    parser.add_argument("--policy", default="identity")
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--tp-size", type=int, default=1)
    return parser


def set_env(args: argparse.Namespace) -> None:
    os.environ["REPLAY_MODE"] = args.mode
    os.environ["REPLAY_PROMPT_TEMPLATE_NAME"] = args.policy
    os.environ["REPLAY_TIMES"] = "1"
    os.environ["REPLAY_IMAGE_COPY_MODE"] = "reuse_path"
    os.environ["REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT"] = "1"
    os.environ.setdefault("REPLAY_LIMIT_MM_PER_PROMPT", "2")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("MINICPM_DEBUG_IO", "1")
    os.environ.setdefault("MINICPM_DEBUG_IO_EVERY", "1")
    os.environ.setdefault("MINICPM_DEBUG_IO_MAX_TEXT_CHARS", "12000")
    os.environ.setdefault("MINICPM_DEBUG_IO_MAX_OUTPUT_CHARS", "12000")
    if args.use_vllm:
        os.environ["MINICPM45_USE_VLLM"] = "1"
        os.environ["MINICPM45_VLLM_TP_SIZE"] = str(args.tp_size)
        os.environ.setdefault("MINICPM45_VLLM_MAX_NUM_SEQS", "2")


def main() -> int:
    args = build_parser().parse_args()
    set_env(args)

    from vlmeval.dataset import build_dataset
    from vlmeval.vlm.minicpm_v_4_5_replay import MiniCPM_V_4_5_Replay

    dataset = build_dataset(args.dataset)
    row = dataset.data.iloc[args.index]
    model = MiniCPM_V_4_5_Replay(
        model_path=args.model_path,
        use_vllm=args.use_vllm,
        tensor_parallel_size=args.tp_size,
    )
    if hasattr(dataset, "dump_image"):
        model.dump_image_func = dataset.dump_image
    message = model.build_prompt(row, dataset=args.dataset)
    response = model.generate_inner(message, dataset=args.dataset)

    payload = {
        "dataset": args.dataset,
        "index": args.index,
        "question_id": row.get("index", row.get("question_id", args.index)),
        "mode": args.mode,
        "policy": args.policy,
        "use_vllm": args.use_vllm,
        "message": message,
        "response": response,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
