#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time


def get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw.isdigit():
        return int(raw)
    return default


def get_env_sizes(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out or default


def build_mathvision_messages(batch_size: int, unique_count: int) -> list[dict]:
    from vlmeval.dataset import build_dataset

    ds = build_dataset("MathVision")
    prompts = []
    for i in range(unique_count):
        prompts.append(ds.build_prompt(ds.data.iloc[i]))
    return [prompts[i % unique_count] for i in range(batch_size)]


def print_summary(kind: str, batch_size: int, elapsed: float, outputs: list[object]) -> dict[str, object]:
    payload = {
        "kind": kind,
        "batch_size": batch_size,
        "elapsed_sec": round(elapsed, 4),
        "samples_per_sec": round(batch_size / elapsed, 4) if elapsed > 0 else None,
        "mean_output_chars": round(sum(len(str(x)) for x in outputs) / max(1, len(outputs)), 2),
        "preview": [str(x)[:120] for x in outputs[:3]],
    }
    return payload


def visible_gpu_indices() -> list[int]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return []
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out


def query_gpu_memory_sum(gpu_indices: list[int]) -> int | None:
    if not gpu_indices:
        return None
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    text = subprocess.check_output(cmd, text=True)
    mem_map: dict[int, int] = {}
    for line in text.strip().splitlines():
        idx_s, mem_s = [x.strip() for x in line.split(",", 1)]
        if idx_s.isdigit() and mem_s.isdigit():
            mem_map[int(idx_s)] = int(mem_s)
    return sum(mem_map.get(idx, 0) for idx in gpu_indices)


def run_with_mem_poll(fn):
    gpu_indices = visible_gpu_indices()
    max_mem = {"value": 0}
    stop = {"flag": False}

    def poll():
        while not stop["flag"]:
            try:
                val = query_gpu_memory_sum(gpu_indices)
                if val is not None and val > max_mem["value"]:
                    max_mem["value"] = val
            except Exception:
                pass
            time.sleep(0.2)

    t = threading.Thread(target=poll, daemon=True)
    t.start()
    start = time.time()
    try:
        outs = fn()
    finally:
        elapsed = time.time() - start
        stop["flag"] = True
        t.join(timeout=1.0)
    return outs, elapsed, max_mem["value"] or None


def run_qwen35(batch_sizes: list[int], max_new_tokens: int) -> None:
    from vlmeval.vlm.qwen35_vl_replay import Qwen35VLChatReplay

    model_path = os.environ["BATCH_SMOKE_MODEL_PATH"]
    model = Qwen35VLChatReplay(model_path, max_new_tokens=max_new_tokens, use_vllm=True)
    orig = model.llm.generate

    def wrapped(reqs, *args, **kwargs):
        print("QWEN35_BATCH_REQS", len(reqs) if isinstance(reqs, list) else 1, flush=True)
        return orig(reqs, *args, **kwargs)

    model.llm.generate = wrapped
    unique_count = get_env_int("MATHVISION_UNIQUE_COUNT", 4)
    for batch_size in batch_sizes:
        messages = build_mathvision_messages(batch_size, unique_count)
        outs, elapsed, max_mem = run_with_mem_poll(lambda: model.generate_batch_inner(messages, dataset="MathVision"))
        payload = print_summary("qwen35", batch_size, elapsed, outs)
        payload["max_mem_mib"] = max_mem
        print("MATHVISION_BATCH_SUMMARY", json.dumps(payload, ensure_ascii=False), flush=True)


def run_qwen2(batch_sizes: list[int], max_new_tokens: int) -> None:
    from vlmeval.vlm.qwen2_vl.model import Qwen2VLChatReplay

    model_path = os.environ["BATCH_SMOKE_MODEL_PATH"]
    model = Qwen2VLChatReplay(model_path, max_new_tokens=max_new_tokens, use_vllm=True)
    orig = model.llm.generate

    def wrapped(reqs, *args, **kwargs):
        print("QWEN2_BATCH_REQS", len(reqs) if isinstance(reqs, list) else 1, flush=True)
        return orig(reqs, *args, **kwargs)

    model.llm.generate = wrapped
    unique_count = get_env_int("MATHVISION_UNIQUE_COUNT", 4)
    for batch_size in batch_sizes:
        messages = build_mathvision_messages(batch_size, unique_count)
        outs, elapsed, max_mem = run_with_mem_poll(lambda: model.generate_batch_inner(messages, dataset="MathVision"))
        payload = print_summary("qwen2", batch_size, elapsed, outs)
        payload["max_mem_mib"] = max_mem
        print("MATHVISION_BATCH_SUMMARY", json.dumps(payload, ensure_ascii=False), flush=True)


def run_minicpm(batch_sizes: list[int], max_new_tokens: int) -> None:
    from vlmeval.vlm.minicpm_v_4_5_replay import MiniCPM_V_4_5_Replay

    model_path = os.environ["BATCH_SMOKE_MODEL_PATH"]
    model = MiniCPM_V_4_5_Replay(model_path, max_new_tokens=max_new_tokens, use_vllm=True)
    orig = model.llm.chat

    def wrapped(conversations, *args, **kwargs):
        print("MINICPM_BATCH_CONVS", len(conversations) if isinstance(conversations, list) else 1, flush=True)
        return orig(conversations, *args, **kwargs)

    model.llm.chat = wrapped
    unique_count = get_env_int("MATHVISION_UNIQUE_COUNT", 4)
    for batch_size in batch_sizes:
        messages = build_mathvision_messages(batch_size, unique_count)
        outs, elapsed, max_mem = run_with_mem_poll(lambda: model.generate_batch_inner(messages, dataset="MathVision"))
        payload = print_summary("minicpm", batch_size, elapsed, outs)
        payload["max_mem_mib"] = max_mem
        print("MATHVISION_BATCH_SUMMARY", json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: probe_mathvision_batch.py [qwen2|qwen35|minicpm]")

    kind = sys.argv[1].strip().lower()
    batch_sizes = get_env_sizes("BATCH_SMOKE_SIZES", [get_env_int("BATCH_SMOKE_SIZE", 4)])
    max_new_tokens = get_env_int("BATCH_SMOKE_MAX_NEW_TOKENS", 64)

    if "BATCH_SMOKE_MODEL_PATH" not in os.environ:
        raise SystemExit("BATCH_SMOKE_MODEL_PATH is required")

    if kind == "qwen35":
        run_qwen35(batch_sizes, max_new_tokens)
        return
    if kind == "qwen2":
        run_qwen2(batch_sizes, max_new_tokens)
        return
    if kind == "minicpm":
        run_minicpm(batch_sizes, max_new_tokens)
        return
    raise SystemExit(f"unknown kind: {kind}")


if __name__ == "__main__":
    main()
