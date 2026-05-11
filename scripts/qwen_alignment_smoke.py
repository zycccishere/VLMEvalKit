#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-check Qwen-family runtime alignment against vendored defaults.")
    parser.add_argument("--family", required=True, choices=["qwen2", "qwen25", "qwen35"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="AI2D_TEST")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--policy", default="identity", choices=["identity", "directly_answer"])
    parser.add_argument("--mode", default="image_text")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-util", type=float, default=0.85)
    return parser


def set_replay_env(args: argparse.Namespace) -> None:
    os.environ["REPLAY_MODE"] = args.mode
    os.environ["REPLAY_PROMPT_TEMPLATE_NAME"] = args.policy
    os.environ["REPLAY_TIMES"] = "1"
    os.environ["REPLAY_IMAGE_COPY_MODE"] = "reuse_path"
    os.environ["REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT"] = "1"
    os.environ["REPLAY_LIMIT_MM_PER_PROMPT"] = "2"
    os.environ["REPLAY_STAGE_DEBUG"] = "0"
    os.environ["REPLAY_PROMPT_AUDIT"] = "0"
    os.environ["VLLM_TP_SIZE"] = str(args.tp_size)
    os.environ["VLLM_MAX_NUM_SEQS"] = str(args.max_num_seqs)
    os.environ["VLLM_MAX_MODEL_LEN"] = str(args.max_model_len)
    os.environ["QWEN35_VLLM_TP_SIZE"] = str(args.tp_size)
    os.environ["QWEN35_VLLM_MAX_NUM_SEQS"] = str(args.max_num_seqs)
    os.environ["QWEN35_VLLM_MAX_MODEL_LEN"] = str(args.max_model_len)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def load_model(args: argparse.Namespace):
    if args.family in {"qwen2", "qwen25"}:
        from vlmeval.vlm.qwen2_vl.model import Qwen2VLChatReplay

        return Qwen2VLChatReplay(
            model_path=args.model_path,
            use_custom_prompt=False,
            use_vllm=True,
            tensor_parallel_size=args.tp_size,
            gpu_utils=args.gpu_util,
        )
    if args.family == "qwen35":
        from vlmeval.vlm.qwen35_vl_replay import Qwen35VLChatReplay

        return Qwen35VLChatReplay(
            model_path=args.model_path,
            use_custom_prompt=False,
            use_vllm=True,
            tensor_parallel_size=args.tp_size,
            gpu_utils=args.gpu_util,
            max_model_len=args.max_model_len,
        )
    raise ValueError(args.family)


def build_message(args: argparse.Namespace):
    from vlmeval.dataset import build_dataset

    dataset = build_dataset(args.dataset)
    row = dataset.data.iloc[args.index]
    message = dataset.build_prompt(row)
    return dataset, row, message


def serialize_sampling(model) -> dict:
    sampling = model._build_vllm_sampling_params()
    fields = [
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "presence_penalty",
        "max_tokens",
        "stop_token_ids",
        "n",
        "best_of",
    ]
    out = {}
    for field in fields:
        if hasattr(sampling, field):
            out[field] = getattr(sampling, field)
    return out


def extract_prompt(req: dict) -> str:
    prompt = req.get("prompt")
    if isinstance(prompt, str):
        return prompt
    return json.dumps(prompt, ensure_ascii=False)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    set_replay_env(args)

    dataset, row, message = build_message(args)
    model = load_model(args)
    req = model._build_vllm_request(message, dataset=args.dataset)
    prompt = extract_prompt(req)
    result = {
        "family": args.family,
        "dataset": args.dataset,
        "policy": args.policy,
        "mode": args.mode,
        "index": int(args.index),
        "question_id": str(row.get("index", row.get("question_id", args.index))),
        "sampling": serialize_sampling(model),
        "prompt_head": prompt[:500],
        "prompt_has_think": "<think>" in prompt,
        "prompt_len": len(prompt),
        "multi_modal_keys": sorted((req.get("multi_modal_data") or {}).keys()),
    }

    if args.family == "qwen35":
        from vlmeval.vlm.qwen35_vl_replay import apply_chat_template_compat, apply_chat_template_nothink

        messages = []
        if getattr(model, "system_prompt", None) is not None:
            messages.append({"role": "system", "content": model.system_prompt})
        messages.append({"role": "user", "content": model._prepare_content(message, dataset=args.dataset)})
        compat = apply_chat_template_compat(model.processor, messages, tokenize=False, add_generation_prompt=True)
        nothink = apply_chat_template_nothink(model.processor, messages, tokenize=False, add_generation_prompt=True)
        active = model._apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result["template_alignment"] = {
            "prefer_direct_answer": bool(model._prefer_direct_answer_mode()),
            "active_matches_compat": active == compat,
            "active_matches_nothink": active == nothink,
        }

    response = model.generate_inner(message, dataset=args.dataset)
    result["response"] = response[:800]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
