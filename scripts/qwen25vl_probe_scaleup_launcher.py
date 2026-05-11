#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
import threading
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "/models/Qwen2.5-VL-7B-Instruct"
DEFAULT_PYTHON = "/opt/miniconda3/envs/vlmevalkit/bin/python"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "probe_scaleup" / "20260320"
BASE_ENV = {
    "PYTHONPATH": ".",
    "VLMEVAL_VLM_MINIMAL_IMPORT": "1",
    "VLMEVAL_API_MINIMAL_IMPORT": "1",
    "VLMEVAL_LAZY_INIT": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
SANITIZED_DISTRIBUTED_ENV_KEYS = (
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the 2026-03-20 Qwen2.5-VL probe scale-up jobs with fair token-length filtering."
    )
    parser.add_argument("--profile", default="scaleup_20260320_5v3_10h")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def base_profile() -> dict:
    standard_root = REPO_ROOT / "runs" / "standard"
    return {
        "name": "scaleup_20260320",
        "jobs": [
            {
                "tag": "image2_dynamath",
                "script": "scripts/qwen25vl_image2_probe.py",
                "dataset": "DynaMath",
                "xlsx_path": str(
                    standard_root
                    / "20260306/qwen2_qwen25_small_newsets_last1_2node16tasks_default_prompt"
                    / "Qwen2.5-VL-7B-Instruct__image_text_image__last1/output/Qwen2VLChatReplay/Qwen2VLChatReplay_DynaMath.xlsx"
                ),
                "policy": "identity",
                "mode": "image_text_image",
                "template_on_last_replay_text": True,
                "min_tokens": 160,
                "sample_count": 1700,
                "gpu_ids": [0, 1, 2, 3, 4],
                "extra_args": ["--max-new-tokens", "160", "--head-reduction", "mean"],
            },
            {
                "tag": "image2_logicvista",
                "script": "scripts/qwen25vl_image2_probe.py",
                "dataset": "LogicVista",
                "xlsx_path": str(
                    standard_root
                    / "20260311/repair_small_reasoning_heavy_1node8tasks_default_prompt"
                    / "Qwen2.5-VL-7B-Instruct__image_text_image__last1/output/Qwen2VLChatReplay/Qwen2VLChatReplay_LogicVista.xlsx"
                ),
                "policy": "identity",
                "mode": "image_text_image",
                "template_on_last_replay_text": True,
                "min_tokens": 160,
                "sample_count": 346,
                "gpu_ids": [5],
                "extra_args": ["--max-new-tokens", "160", "--head-reduction", "mean"],
            },
            {
                "tag": "image2_seedbench",
                "script": "scripts/qwen25vl_image2_probe.py",
                "dataset": "SEEDBench2_Plus",
                "xlsx_path": str(
                    standard_root
                    / "20260306/qwen2_qwen25_small_newsets_last1_2node16tasks_default_prompt"
                    / "Qwen2.5-VL-7B-Instruct__image_text_image__last1/output/Qwen2VLChatReplay/Qwen2VLChatReplay_SEEDBench2_Plus.xlsx"
                ),
                "policy": "directly_answer",
                "mode": "image_text_image",
                "template_on_last_replay_text": True,
                "min_tokens": 12,
                "sample_count": 244,
                "gpu_ids": [6],
                "extra_args": ["--max-new-tokens", "12", "--head-reduction", "mean"],
            },
            {
                "tag": "cache_swap_dynamath",
                "script": "scripts/qwen25vl_cache_swap_probe.py",
                "dataset": "DynaMath",
                "xlsx_path": str(
                    standard_root
                    / "20260306/qwen2_qwen25_small_newsets_last1_2node16tasks_default_prompt"
                    / "Qwen2.5-VL-7B-Instruct__image_text_image__last1/output/Qwen2VLChatReplay/Qwen2VLChatReplay_DynaMath.xlsx"
                ),
                "policy": "identity",
                "mode": "image_text_image",
                "template_on_last_replay_text": True,
                "min_tokens": 160,
                "sample_count": 160,
                "gpu_ids": [7],
                "extra_args": ["--teacher-force-steps", "96", "--skip-short-samples", "--head-reduction", "mean"],
            },
            {
                "tag": "cache_swap_logicvista",
                "script": "scripts/qwen25vl_cache_swap_probe.py",
                "dataset": "LogicVista",
                "xlsx_path": str(
                    standard_root
                    / "20260311/repair_small_reasoning_heavy_1node8tasks_default_prompt"
                    / "Qwen2.5-VL-7B-Instruct__image_text_image__last1/output/Qwen2VLChatReplay/Qwen2VLChatReplay_LogicVista.xlsx"
                ),
                "policy": "identity",
                "mode": "image_text_image",
                "template_on_last_replay_text": True,
                "min_tokens": 160,
                "sample_count": 128,
                "gpu_ids": [7],
                "extra_args": ["--teacher-force-steps", "96", "--skip-short-samples", "--head-reduction", "mean"],
            },
            {
                "tag": "cache_swap_seedbench",
                "script": "scripts/qwen25vl_cache_swap_probe.py",
                "dataset": "SEEDBench2_Plus",
                "xlsx_path": str(
                    standard_root
                    / "20260306/qwen2_qwen25_small_newsets_last1_2node16tasks_default_prompt"
                    / "Qwen2.5-VL-7B-Instruct__image_text_image__last1/output/Qwen2VLChatReplay/Qwen2VLChatReplay_SEEDBench2_Plus.xlsx"
                ),
                "policy": "directly_answer",
                "mode": "image_text_image",
                "template_on_last_replay_text": True,
                "min_tokens": 12,
                "sample_count": 96,
                "gpu_ids": [7],
                "extra_args": ["--teacher-force-steps", "12", "--skip-short-samples", "--head-reduction", "mean"],
            },
        ],
    }


def profile_5v3_10h() -> dict:
    standard_root = REPO_ROOT / "runs" / "standard"
    return {
        "name": "scaleup_20260320_5v3_10h",
        "jobs": [
            {
                "tag": "image2_dynamath",
                "script": "scripts/qwen25vl_image2_probe.py",
                "dataset": "DynaMath",
                "xlsx_path": str(
                    standard_root
                    / "20260306/qwen2_qwen25_small_newsets_last1_2node16tasks_default_prompt"
                    / "Qwen2.5-VL-7B-Instruct__image_text_image__last1/output/Qwen2VLChatReplay/Qwen2VLChatReplay_DynaMath.xlsx"
                ),
                "policy": "identity",
                "mode": "image_text_image",
                "template_on_last_replay_text": True,
                "min_tokens": 160,
                "sample_count": 1500,
                "gpu_ids": [0, 1, 2, 3],
                "extra_args": ["--max-new-tokens", "160", "--head-reduction", "mean"],
            },
            {
                "tag": "image2_logicvista",
                "script": "scripts/qwen25vl_image2_probe.py",
                "dataset": "LogicVista",
                "xlsx_path": str(
                    standard_root
                    / "20260311/repair_small_reasoning_heavy_1node8tasks_default_prompt"
                    / "Qwen2.5-VL-7B-Instruct__image_text_image__last1/output/Qwen2VLChatReplay/Qwen2VLChatReplay_LogicVista.xlsx"
                ),
                "policy": "identity",
                "mode": "image_text_image",
                "template_on_last_replay_text": True,
                "min_tokens": 160,
                "sample_count": 346,
                "gpu_ids": [4],
                "extra_args": ["--max-new-tokens", "160", "--head-reduction", "mean"],
            },
            {
                "tag": "image2_seedbench",
                "script": "scripts/qwen25vl_image2_probe.py",
                "dataset": "SEEDBench2_Plus",
                "xlsx_path": str(
                    standard_root
                    / "20260306/qwen2_qwen25_small_newsets_last1_2node16tasks_default_prompt"
                    / "Qwen2.5-VL-7B-Instruct__image_text_image__last1/output/Qwen2VLChatReplay/Qwen2VLChatReplay_SEEDBench2_Plus.xlsx"
                ),
                "policy": "directly_answer",
                "mode": "image_text_image",
                "template_on_last_replay_text": True,
                "min_tokens": 12,
                "sample_count": 244,
                "gpu_ids": [4],
                "extra_args": ["--max-new-tokens", "12", "--head-reduction", "mean"],
            },
            {
                "tag": "cache_swap_dynamath",
                "script": "scripts/qwen25vl_cache_swap_probe.py",
                "dataset": "DynaMath",
                "xlsx_path": str(
                    standard_root
                    / "20260306/qwen2_qwen25_small_newsets_last1_2node16tasks_default_prompt"
                    / "Qwen2.5-VL-7B-Instruct__image_text_image__last1/output/Qwen2VLChatReplay/Qwen2VLChatReplay_DynaMath.xlsx"
                ),
                "policy": "identity",
                "mode": "image_text_image",
                "template_on_last_replay_text": True,
                "min_tokens": 160,
                "sample_count": 1600,
                "gpu_ids": [5, 6],
                "extra_args": ["--teacher-force-steps", "96", "--skip-short-samples", "--head-reduction", "mean"],
            },
            {
                "tag": "cache_swap_logicvista",
                "script": "scripts/qwen25vl_cache_swap_probe.py",
                "dataset": "LogicVista",
                "xlsx_path": str(
                    standard_root
                    / "20260311/repair_small_reasoning_heavy_1node8tasks_default_prompt"
                    / "Qwen2.5-VL-7B-Instruct__image_text_image__last1/output/Qwen2VLChatReplay/Qwen2VLChatReplay_LogicVista.xlsx"
                ),
                "policy": "identity",
                "mode": "image_text_image",
                "template_on_last_replay_text": True,
                "min_tokens": 160,
                "sample_count": 346,
                "gpu_ids": [7],
                "extra_args": ["--teacher-force-steps", "160", "--skip-short-samples", "--head-reduction", "mean"],
            },
            {
                "tag": "cache_swap_seedbench",
                "script": "scripts/qwen25vl_cache_swap_probe.py",
                "dataset": "SEEDBench2_Plus",
                "xlsx_path": str(
                    standard_root
                    / "20260306/qwen2_qwen25_small_newsets_last1_2node16tasks_default_prompt"
                    / "Qwen2.5-VL-7B-Instruct__image_text_image__last1/output/Qwen2VLChatReplay/Qwen2VLChatReplay_SEEDBench2_Plus.xlsx"
                ),
                "policy": "directly_answer",
                "mode": "image_text_image",
                "template_on_last_replay_text": True,
                "min_tokens": 12,
                "sample_count": 244,
                "gpu_ids": [7],
                "extra_args": ["--teacher-force-steps", "12", "--skip-short-samples", "--head-reduction", "mean"],
            },
        ],
    }


PROFILES = {
    "scaleup_20260320": base_profile,
    "scaleup_20260320_5v3_10h": profile_5v3_10h,
}


def prediction_token_lengths(xlsx_path: str, tokenizer) -> list[int]:
    df = pd.read_excel(xlsx_path)
    predictions = df["prediction"].fillna("").astype(str).tolist()
    return [len(tokenizer.encode(text, add_special_tokens=False)) for text in predictions]


def select_indices(job_cfg: dict, *, tokenizer, seed: int) -> dict:
    lengths = prediction_token_lengths(job_cfg["xlsx_path"], tokenizer)
    pool = [int(idx) for idx, token_len in enumerate(lengths) if token_len >= job_cfg["min_tokens"]]
    rng = random.Random(seed)
    selected_count = min(job_cfg["sample_count"], len(pool))
    selected = rng.sample(pool, selected_count)
    return {
        "pool_size": len(pool),
        "selected_count": selected_count,
        "selected_indices": selected,
        "length_threshold": job_cfg["min_tokens"],
    }


def shard_indices(indices: list[int], shard_count: int) -> list[list[int]]:
    shards = [[] for _ in range(shard_count)]
    for idx, value in enumerate(indices):
        shards[idx % shard_count].append(value)
    return shards


def build_job_plan(profile_name: str, *, seed: int, tokenizer) -> dict:
    profile = deepcopy(PROFILES[profile_name]())
    jobs = []
    for offset, job_cfg in enumerate(profile["jobs"]):
        selection = select_indices(job_cfg, tokenizer=tokenizer, seed=seed + offset)
        gpu_ids = job_cfg["gpu_ids"]
        shards = shard_indices(selection["selected_indices"], len(gpu_ids))
        job_cfg["selection"] = selection
        job_cfg["seed"] = seed + offset
        job_cfg["shards"] = [
            {
                "gpu_id": gpu_id,
                "indices": shard_indices_one,
                "shard_id": shard_id,
            }
            for shard_id, (gpu_id, shard_indices_one) in enumerate(zip(gpu_ids, shards))
            if shard_indices_one
        ]
        jobs.append(job_cfg)
    profile["jobs"] = jobs
    return profile


def run_gpu_queue(
    gpu_id: int,
    queue: list[dict],
    *,
    python_bin: str,
    output_root: Path,
    dry_run: bool,
) -> None:
    env = os.environ.copy()
    env.update(BASE_ENV)
    for key in SANITIZED_DISTRIBUTED_ENV_KEYS:
        env.pop(key, None)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    for job in queue:
        shard_output = output_root / job["output_subdir"]
        shard_output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "profile": job["profile"],
            "tag": job["tag"],
            "dataset": job["dataset"],
            "policy": job["policy"],
            "mode": job["mode"],
            "gpu_id": gpu_id,
            "seed": job["seed"],
            "xlsx_path": job["xlsx_path"],
            "length_threshold": job["selection"]["length_threshold"],
            "pool_size": job["selection"]["pool_size"],
            "selected_count_total": job["selection"]["selected_count"],
            "shard_id": job["shard_id"],
            "shard_count": job["shard_count"],
            "indices": job["indices"],
            "extra_args": job["extra_args"],
        }
        (shard_output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        cmd = [
            python_bin,
            job["script"],
            "--dataset",
            job["dataset"],
            "--indices",
            *[str(x) for x in job["indices"]],
            "--mode",
            job["mode"],
            "--policy",
            job["policy"],
            "--output-dir",
            str(shard_output),
            *job["extra_args"],
        ]
        if job["template_on_last_replay_text"]:
            cmd.append("--template-on-last-replay-text")

        rendered = " ".join(shlex.quote(part) for part in cmd)
        print(f"[gpu{gpu_id}] {job['tag']} shard {job['shard_id']}: {rendered}", flush=True)
        if dry_run:
            continue

        start_time = time.time()
        with (shard_output / "run.log").open("w", encoding="utf-8") as log_fp:
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
            )
            rc = proc.wait()
        duration = time.time() - start_time
        result = {"returncode": rc, "duration_seconds": duration}
        (shard_output / "launcher_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if rc != 0:
            raise RuntimeError(f"Job failed on gpu{gpu_id}: {job['tag']} shard {job['shard_id']} rc={rc}")


def main() -> int:
    args = build_parser().parse_args()
    if args.profile not in PROFILES:
        raise ValueError(f"Unknown profile: {args.profile}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    output_root = Path(args.output_root) / args.profile
    output_root.mkdir(parents=True, exist_ok=True)

    profile = build_job_plan(args.profile, seed=args.seed, tokenizer=tokenizer)
    (output_root / "launch_plan.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    jobs_by_gpu: dict[int, list[dict]] = defaultdict(list)
    for job_cfg in profile["jobs"]:
        for shard in job_cfg["shards"]:
            jobs_by_gpu[shard["gpu_id"]].append(
                {
                    "profile": profile["name"],
                    "tag": job_cfg["tag"],
                    "script": job_cfg["script"],
                    "dataset": job_cfg["dataset"],
                    "policy": job_cfg["policy"],
                    "mode": job_cfg["mode"],
                    "template_on_last_replay_text": job_cfg["template_on_last_replay_text"],
                    "extra_args": job_cfg["extra_args"],
                    "indices": shard["indices"],
                    "seed": job_cfg["seed"],
                    "xlsx_path": job_cfg["xlsx_path"],
                    "selection": job_cfg["selection"],
                    "shard_id": shard["shard_id"],
                    "shard_count": len(job_cfg["shards"]),
                    "output_subdir": f"{job_cfg['tag']}/gpu{shard['gpu_id']}_shard{shard['shard_id']}",
                }
            )

    threads = []
    errors: list[str] = []

    def worker(gpu_id: int, queue: list[dict]) -> None:
        try:
            run_gpu_queue(
                gpu_id,
                queue,
                python_bin=args.python_bin,
                output_root=output_root,
                dry_run=args.dry_run,
            )
        except Exception as exc:  # pragma: no cover
            errors.append(f"gpu{gpu_id}: {exc}")

    for gpu_id, queue in sorted(jobs_by_gpu.items()):
        thread = threading.Thread(target=worker, args=(gpu_id, queue), daemon=False)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    if errors:
        raise RuntimeError("; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
