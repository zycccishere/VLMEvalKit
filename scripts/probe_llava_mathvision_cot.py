#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path


DEFAULT_COT_PREFIX = (
    "Read the following question carefully, solve it step by step, and then "
    'output the final answer in the format of "Answer: single number or single word or phrase".'
)


def configure_env(model_path: str, mode: str, policy: str, use_vllm: bool) -> None:
    os.environ["MODEL_PATH"] = model_path
    os.environ["REPLAY_MODE"] = mode
    os.environ["REPLAY_PROMPT_TEMPLATE_NAME"] = policy
    os.environ["REPLAY_TIMES"] = "1"
    os.environ["REPLAY_IMAGE_COPY_MODE"] = "reuse_path"
    os.environ["REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["LLAVA_USE_VLLM"] = "1" if use_vllm else "0"


def build_message(model, dataset, row):
    if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(dataset.dataset_name):
        return model.build_prompt(row, dataset=dataset.dataset_name)
    return dataset.build_prompt(row)


def add_cot_prefix(message, prefix: str):
    msg = copy.deepcopy(message)
    for item in msg:
        if item.get("type") == "text":
            item["value"] = f"{prefix}\n\n{item['value']}"
            return msg
    raise ValueError("No text item found in message")


def evaluate_mathvision_line(row, prediction, judge_model):
    from vlmeval.dataset.utils.mathv import MATH_V_auxeval, post_check

    line = row.to_dict()
    line["prediction"] = prediction
    aux = MATH_V_auxeval(judge_model, line)
    line["res"] = aux["res"]
    hit = bool(post_check(line, prefetch=False))
    return {
        "log": aux["log"],
        "res": aux["res"],
        "hit": int(hit),
    }


def evaluate_mathvision_line_local_only(row, prediction):
    from vlmeval.dataset.utils.mathv import post_check

    line = row.to_dict()
    line["prediction"] = prediction
    pref = post_check(line, prefetch=True)
    line["res"] = pref if pref else ""
    hit = bool(post_check(line, prefetch=False)) if pref else False
    return {
        "log": "Local prefetch succeed" if pref else "Local unresolved",
        "res": line["res"],
        "hit": int(hit),
        "unresolved": int(not pref),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare LLaVA-1.5 MathVision accuracy with/without CoT prefix.")
    parser.add_argument("--model-path", default="/models/llava-1.5-7b-hf")
    parser.add_argument("--registry-name", default="llava_v1.5_7b_hf_replay")
    parser.add_argument("--dataset", default="MathVision")
    parser.add_argument("--mode", default="image_text")
    parser.add_argument("--policy", default="identity")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--cot-prefix", default=DEFAULT_COT_PREFIX)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    configure_env(args.model_path, args.mode, args.policy, args.use_vllm)

    from vlmeval.config_runtime import supported_VLM
    from vlmeval.dataset import build_dataset
    from vlmeval.dataset.utils.judge_util import build_judge

    dataset = build_dataset(args.dataset)
    if dataset is None:
        raise ValueError(f"Failed to build dataset: {args.dataset}")

    model = supported_VLM[args.registry_name]()
    model.set_dump_image(dataset.dump_image)
    judge = None if args.local_only else build_judge(model=args.judge_model)

    end_index = min(args.start_index + args.sample_count, len(dataset.data))
    rows = dataset.data.iloc[args.start_index:end_index]

    results = []
    no_cot_hits = 0
    cot_hits = 0
    no_cot_unresolved = 0
    cot_unresolved = 0
    for _, row in rows.iterrows():
        no_cot_msg = build_message(model, dataset, row)
        cot_msg = add_cot_prefix(no_cot_msg, args.cot_prefix)

        no_cot_pred = model.generate(message=no_cot_msg, dataset=dataset.dataset_name)
        cot_pred = model.generate(message=cot_msg, dataset=dataset.dataset_name)

        if args.local_only:
            no_cot_eval = evaluate_mathvision_line_local_only(row, no_cot_pred)
            cot_eval = evaluate_mathvision_line_local_only(row, cot_pred)
        else:
            no_cot_eval = evaluate_mathvision_line(row, no_cot_pred, judge)
            cot_eval = evaluate_mathvision_line(row, cot_pred, judge)

        no_cot_hits += no_cot_eval["hit"]
        cot_hits += cot_eval["hit"]
        no_cot_unresolved += int(no_cot_eval.get("unresolved", 0))
        cot_unresolved += int(cot_eval.get("unresolved", 0))

        text_prompt = next(item["value"] for item in no_cot_msg if item["type"] == "text")
        cot_text_prompt = next(item["value"] for item in cot_msg if item["type"] == "text")

        results.append(
            {
                "index": str(row["index"]),
                "answer": str(row["answer"]),
                "question_preview": str(row["question"])[:240],
                "no_cot_prompt_preview": text_prompt[:240],
                "cot_prompt_preview": cot_text_prompt[:240],
                "no_cot_prediction": str(no_cot_pred),
                "no_cot_res": str(no_cot_eval["res"]),
                "no_cot_log": str(no_cot_eval["log"]),
                "no_cot_hit": no_cot_eval["hit"],
                "no_cot_unresolved": int(no_cot_eval.get("unresolved", 0)),
                "cot_prediction": str(cot_pred),
                "cot_res": str(cot_eval["res"]),
                "cot_log": str(cot_eval["log"]),
                "cot_hit": cot_eval["hit"],
                "cot_unresolved": int(cot_eval.get("unresolved", 0)),
            }
        )

    total = len(results)
    payload = {
        "dataset": args.dataset,
        "mode": args.mode,
        "policy": args.policy,
        "sample_count": total,
        "start_index": args.start_index,
        "model_path": args.model_path,
        "registry_name": args.registry_name,
        "judge_model": args.judge_model,
        "local_only": args.local_only,
        "use_vllm": args.use_vllm,
        "cot_prefix": args.cot_prefix,
        "no_cot_accuracy": (no_cot_hits / total) if total else 0.0,
        "cot_accuracy": (cot_hits / total) if total else 0.0,
        "no_cot_hits": no_cot_hits,
        "cot_hits": cot_hits,
        "no_cot_unresolved": no_cot_unresolved,
        "cot_unresolved": cot_unresolved,
        "results": results,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
