#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


DEFAULT_DATASETS = [
    "AI2D_TEST",
    "MathVista_MINI",
    "OCRBench",
    "SEEDBench2_Plus",
    "VisuLogic",
    "LogicVista",
    "VisualPuzzles",
    "DynaMath",
    "MathVision",
]


def configure_env(model_path: str, mode: str, policy: str) -> None:
    os.environ["MODEL_PATH"] = model_path
    os.environ["REPLAY_MODE"] = mode
    os.environ["REPLAY_PROMPT_TEMPLATE_NAME"] = policy
    os.environ["REPLAY_TIMES"] = "1"
    os.environ["REPLAY_IMAGE_COPY_MODE"] = "reuse_path"
    os.environ["REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["LLAVA_USE_VLLM"] = "0"


def build_message(model, dataset, row):
    if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(dataset.dataset_name):
        return model.build_prompt(row, dataset=dataset.dataset_name)
    return dataset.build_prompt(row)


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.2f}h"


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate LLaVA-1.5 HF runtime on active benchmarks.")
    parser.add_argument("--model-path", default="/models/llava-1.5-7b-hf")
    parser.add_argument("--registry-name", default="llava_v1.5_7b_hf_replay")
    parser.add_argument("--mode", default="image_text")
    parser.add_argument("--policy", default="identity")
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    configure_env(args.model_path, args.mode, args.policy)

    from vlmeval.config_runtime import supported_VLM
    from vlmeval.dataset import build_dataset

    t0 = time.perf_counter()
    model = supported_VLM[args.registry_name]()
    load_seconds = time.perf_counter() - t0

    results = {
        "model_path": args.model_path,
        "registry_name": args.registry_name,
        "mode": args.mode,
        "policy": args.policy,
        "sample_count": args.sample_count,
        "model_load_seconds": load_seconds,
        "datasets": [],
    }

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    for name in datasets:
        ds = build_dataset(name)
        if ds is None:
            results["datasets"].append({
                "dataset": name,
                "error": "build_dataset returned None",
            })
            continue

        model.set_dump_image(ds.dump_image)
        sample_n = min(args.sample_count, len(ds.data))
        rows = ds.data.iloc[:sample_n]

        prompt_seconds = 0.0
        infer_seconds = 0.0
        sample_outputs = []

        for _, row in rows.iterrows():
            p0 = time.perf_counter()
            message = build_message(model, ds, row)
            prompt_seconds += time.perf_counter() - p0

            i0 = time.perf_counter()
            output = model.generate(message=message, dataset=ds.dataset_name)
            infer_seconds += time.perf_counter() - i0

            if len(sample_outputs) < 2:
                sample_outputs.append(str(output)[:200])

        avg_infer = infer_seconds / sample_n if sample_n else 0.0
        avg_prompt = prompt_seconds / sample_n if sample_n else 0.0
        estimated_total = load_seconds + avg_infer * len(ds.data)
        results["datasets"].append({
            "dataset": ds.dataset_name,
            "dataset_size": int(len(ds.data)),
            "sampled": int(sample_n),
            "avg_prompt_seconds": avg_prompt,
            "avg_infer_seconds": avg_infer,
            "estimated_total_seconds": estimated_total,
            "estimated_total_human": format_seconds(estimated_total),
            "sample_outputs": sample_outputs,
        })

    payload = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
