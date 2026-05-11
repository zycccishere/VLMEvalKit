#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_ROOT = SCRIPT_DIR / "configs" / "rope_probe_20260330"
REMOTE_VLMEVAL_ROOT = Path("/path/to/vlmevalkit")
REMOTE_LLAVA_ROOT = Path("/path/to/LLaVA")
REMOTE_VLMEVAL_PY = "/opt/miniconda3/envs/vlmevalkit/bin/python"
REMOTE_LLAVA_PY = REMOTE_VLMEVAL_PY


@dataclass(frozen=True)
class ModelSpec:
    key: str
    kind: str
    model_path: str
    python_bin: str
    repo_root: Path
    script_path: Path
    gpus_per_job: int
    decode_profile_path: Path
    min_free_memory_mb: int


def split_csv(raw: str) -> list[str]:
    if not raw:
        return []
    return [part for part in raw.replace(",", " ").split() if part]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ceil_scaled_estimate(base_minutes: int, sample_count: int) -> int:
    return max(1, round(base_minutes * sample_count / 200.0))


def build_model_specs() -> dict[str, ModelSpec]:
    shared_llava_profile = CONFIG_ROOT / "decode_profile_llava13b_aligned_shared_image_text_image.json"
    return {
        "qwen25vl_32b": ModelSpec(
            key="qwen25vl_32b",
            kind="qwen",
            model_path="/models/Qwen2.5-VL-32B-Instruct",
            python_bin=REMOTE_VLMEVAL_PY,
            repo_root=REMOTE_VLMEVAL_ROOT,
            script_path=REMOTE_VLMEVAL_ROOT / "scripts" / "qwen25vl_image2_probe.py",
            gpus_per_job=2,
            decode_profile_path=CONFIG_ROOT / "decode_profile_qwen25vl_32b_image_text_image.json",
            min_free_memory_mb=60000,
        ),
        "llava_aligned_iti": ModelSpec(
            key="llava_aligned_iti",
            kind="llava",
            model_path="/path/to/LLaVA/checkpoints/llava-aligned-pt-image_text_image-sft-image_text_image",
            python_bin=REMOTE_LLAVA_PY,
            repo_root=REMOTE_LLAVA_ROOT,
            script_path=REMOTE_LLAVA_ROOT / "scripts" / "llava_image2_rope_probe.py",
            gpus_per_job=1,
            decode_profile_path=shared_llava_profile,
            min_free_memory_mb=30000,
        ),
        "llava_aligned_it": ModelSpec(
            key="llava_aligned_it",
            kind="llava",
            model_path="/path/to/LLaVA/checkpoints/llava-aligned-pt-image_text-sft-image_text",
            python_bin=REMOTE_LLAVA_PY,
            repo_root=REMOTE_LLAVA_ROOT,
            script_path=REMOTE_LLAVA_ROOT / "scripts" / "llava_image2_rope_probe.py",
            gpus_per_job=1,
            decode_profile_path=shared_llava_profile,
            min_free_memory_mb=30000,
        ),
    }


def manifest_path_for_dataset(dataset: str) -> Path:
    mapping = {
        "AI2D_TEST": CONFIG_ROOT / "manifest_ai2d_test_200_seed42.json",
        "LogicVista": CONFIG_ROOT / "manifest_logicvista_200_seed42.json",
        "MathVision": CONFIG_ROOT / "manifest_mathvision_200_seed42.json",
        "DynaMath": CONFIG_ROOT / "manifest_dynamath_200_first_two_shards_20260325.json",
    }
    if dataset not in mapping:
        raise KeyError(f"Unsupported dataset: {dataset}")
    return mapping[dataset]


def indices_for_dataset(dataset: str, *, max_samples: int) -> list[int]:
    manifest = load_json(manifest_path_for_dataset(dataset))
    indices = list(manifest["indices"])
    if max_samples > 0:
        indices = indices[:max_samples]
    return indices


def max_new_tokens(spec: ModelSpec, dataset: str) -> int:
    profile = load_json(spec.decode_profile_path)
    return int(profile["datasets"][dataset]["max_new_tokens"])


def estimated_minutes_for_dataset(dataset: str, sample_count: int) -> int:
    base = {
        "AI2D_TEST": 30,
        "LogicVista": 50,
        "MathVision": 95,
        "DynaMath": 55,
    }[dataset]
    return ceil_scaled_estimate(base, sample_count)


def build_job_env(spec: ModelSpec) -> dict[str, str]:
    env: dict[str, str] = {}
    if spec.kind == "qwen":
        env["PYTHONPATH"] = str(REMOTE_VLMEVAL_ROOT)
        env["VLMEVAL_VLM_MINIMAL_IMPORT"] = "1"
        env["VLMEVAL_API_MINIMAL_IMPORT"] = "1"
        env["VLMEVAL_LAZY_INIT"] = "1"
    else:
        env["PYTHONPATH"] = ":".join([str(REMOTE_LLAVA_ROOT), str(REMOTE_VLMEVAL_ROOT)])
        env["VLMEVAL_ROOT"] = str(REMOTE_VLMEVAL_ROOT)
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
        env["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"
    return env


def build_command(
    *,
    spec: ModelSpec,
    dataset: str,
    condition: str,
    indices: list[int],
    output_root: Path,
    attn_layers: str,
    head_reduction: str,
) -> list[str]:
    output_dir = output_root / dataset / spec.key / condition
    cmd = [
        spec.python_bin,
        str(spec.script_path),
        "--model-path",
        spec.model_path,
        "--dataset",
        dataset,
        "--indices",
        *[str(i) for i in indices],
        "--mode",
        "image_text_image",
        "--policy",
        "identity",
        "--template-on-last-replay-text",
        "--max-new-tokens",
        str(max_new_tokens(spec, dataset)),
        "--attn-layers",
        attn_layers,
        "--head-reduction",
        head_reduction,
        "--output-dir",
        str(output_dir),
    ]
    if spec.kind == "qwen":
        cmd.extend(["--device", "auto"])
    if condition == "rope_align":
        cmd.append("--rope-align")
    return cmd
