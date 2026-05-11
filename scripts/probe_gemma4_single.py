#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe one Gemma4 replay sample.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="AI2D_TEST")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--mode", default="image_text_image")
    parser.add_argument("--policy", default="directly_answer", choices=["identity", "directly_answer"])
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--gpu-util", type=float, default=0.85)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser


def set_env(args: argparse.Namespace) -> None:
    os.environ["REPLAY_MODE"] = args.mode
    os.environ["REPLAY_PROMPT_TEMPLATE_NAME"] = args.policy
    os.environ["REPLAY_TIMES"] = "1"
    os.environ["REPLAY_IMAGE_COPY_MODE"] = "reuse_path"
    os.environ["REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT"] = "1"
    os.environ.setdefault("VLMEVAL_USE_GEMMA4_MINIMAL_CONFIG", "1")
    os.environ.setdefault("VLMEVAL_VLM_MINIMAL_IMPORT", "1")
    os.environ.setdefault("VLMEVAL_API_MINIMAL_IMPORT", "1")
    os.environ.setdefault("VLMEVAL_LAZY_INIT", "1")
    os.environ.setdefault("REPLAY_LIMIT_MM_PER_PROMPT", "2")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["GEMMA4_ENABLE_THINKING"] = "1" if args.enable_thinking else "0"
    if args.use_vllm:
        os.environ["GEMMA4_USE_VLLM"] = "1"
        os.environ["GEMMA4_VLLM_TP_SIZE"] = str(args.tp_size)
        os.environ["GEMMA4_VLLM_MAX_MODEL_LEN"] = str(args.max_model_len)
        os.environ["GEMMA4_VLLM_MAX_NUM_SEQS"] = str(args.max_num_seqs)


def main() -> int:
    args = build_parser().parse_args()
    set_env(args)

    from vlmeval.dataset import build_dataset
    from vlmeval.vlm.gemma4_replay import Gemma4Replay

    dataset = build_dataset(args.dataset)
    row = dataset.data.iloc[args.index]
    model = Gemma4Replay(
        model_path=args.model_path,
        use_vllm=args.use_vllm,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_utils=args.gpu_util,
    )
    if hasattr(dataset, "dump_image"):
        model.set_dump_image(dataset.dump_image)
    message = model.build_prompt(row, dataset=args.dataset)
    response = model.generate_inner(message, dataset=args.dataset)

    payload = {
        "dataset": args.dataset,
        "index": args.index,
        "question_id": row.get("index", row.get("question_id", args.index)),
        "mode": args.mode,
        "policy": args.policy,
        "use_vllm": args.use_vllm,
        "enable_thinking": args.enable_thinking,
        "message": message,
        "response": response,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
