from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from unittest import mock


REPLAY_MODES = {
    "IQ": "image_text",
    "QI": "text_image",
    "IQQ": "image_text_text",
    "IIQ": "image_image_text",
    "IQI": "image_text_image",
    "IQIQ": "image_text_image_text",
}
EXPECTED_IMAGE_COUNTS = {
    "image_text": 1,
    "text_image": 1,
    "image_text_text": 1,
    "image_image_text": 2,
    "image_text_image": 2,
    "image_text_image_text": 2,
}
LOGICVISTA_QWEN_SAMPLING = {
    "VLLM_USE_V1": "0",
    "LOGICVISTA_QWEN25VL_LEGACY_SAMPLING": "1",
    "QWEN2VL_VLLM_REPETITION_PENALTY": "1.05",
    "QWEN2VL_VLLM_TEMPERATURE": "0.01",
    "QWEN2VL_VLLM_TOP_P": "1.0",
    "QWEN2VL_VLLM_TOP_K": "0",
    "QWEN2VL_VLLM_MAX_TOKENS": "2048",
    "QWEN2VL_VLLM_STOP_TOKEN_IDS": "151645,151643",
}


class ProbeRegistry:
    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    def register(self, name: str):
        def deco(func):
            if name in self._items:
                raise KeyError(f"duplicate probe: {name}")
            self._items[name] = func
            return func
        return deco

    def get(self, name: str):
        if name not in self._items:
            raise KeyError(f"unknown probe: {name}")
        return self._items[name]


REGISTRY = ProbeRegistry()


def json_default(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    try:
        import pandas as pd
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


@contextlib.contextmanager
def patched_environ(env: dict[str, str]):
    old = os.environ.copy()
    os.environ.clear()
    os.environ.update(old)
    os.environ.update({str(k): str(v) for k, v in env.items()})
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def split_csv(raw: str) -> list[str]:
    return [x for x in raw.replace(",", " ").split() if x]



ACTIVE_MODEL_PATHS = {
    "qwen25vl_3b": "Qwen2.5-VL-3B-Instruct",
    "qwen25vl_7b": "Qwen2.5-VL-7B-Instruct",
    "qwen25vl_32b": "Qwen2.5-VL-32B-Instruct",
    "qwen25vl_72b": "Qwen2.5-VL-72B-Instruct",
    "minicpm_v_45": "MiniCPM-V-4_5",
    "minicpm_v_45_no_reasoning": "MiniCPM-V-4_5",
    "minicpm_v_45_regular_reasoning": "MiniCPM-V-4_5",
    "minicpm_o_45": "MiniCPM-o-4_5",
    "minicpm_o_45_no_reasoning": "MiniCPM-o-4_5",
    "minicpm_o_45_regular_reasoning": "MiniCPM-o-4_5",
    "gemma3_4b": "gemma-3-4b-it",
    "gemma3_12b": "gemma-3-12b-it",
    "gemma3_27b": "gemma-3-27b-it",
}
ACTIVE_MODEL_KEYS = [
    "qwen25vl_3b", "qwen25vl_7b", "qwen25vl_32b", "qwen25vl_72b",
    "minicpm_v_45", "minicpm_o_45", "gemma3_4b", "gemma3_12b", "gemma3_27b",
]
API_MODEL_KEYS = [
    "gpt4o_mini", "gpt_4o_mini", "gpt_5_mini", "gpt_5_2025_08_07", "gpt_5_chat",
    "claude_haiku_4_5_20251001", "gemini_25_flash_lite", "gemini_25_flash_nothinking",
    "gemini_25_flash_thinking", "gemini_3_flash_preview_nothinking", "gemini_31_flash_lite",
]


def model_family(model_key: str) -> str:
    if model_key.startswith("qwen25vl_"):
        return "qwen25vl"
    if model_key.startswith("minicpm_"):
        return "minicpm45"
    if model_key.startswith("gemma3_"):
        return "gemma3"
    return "api"


def resolved_model_path(model_key: str, model_root: str = "/user/zyc1781/models", model_cp_root: str = "/user/zyc1781/models-cp") -> str:
    name = ACTIVE_MODEL_PATHS.get(model_key)
    if not name:
        return ""
    primary = Path(model_root) / name
    if primary.exists():
        return str(primary)
    backup = Path(model_cp_root) / name
    if backup.exists():
        return str(backup)
    return str(primary)


def apply_model_path_override(env: dict[str, str], model_key: str, path_value: str) -> None:
    env["MODEL_PATH"] = path_value
    family = model_family(model_key)
    if family == "qwen25vl":
        if model_key.endswith("_3b"):
            env["MODEL_PATH_QWEN25_3B"] = path_value
        elif model_key.endswith("_7b"):
            env["MODEL_PATH_QWEN25"] = path_value
        elif model_key.endswith("_32b"):
            env["MODEL_PATH_QWEN25_32B"] = path_value
        elif model_key.endswith("_72b"):
            env["MODEL_PATH_QWEN25_72B"] = path_value
    elif model_key.startswith("minicpm_v_"):
        env["MODEL_PATH_MINICPM45"] = path_value
    elif model_key.startswith("minicpm_o_"):
        env["MODEL_PATH_MINICPMO45"] = path_value
    elif model_key == "gemma3_4b":
        env["MODEL_PATH_GEMMA3_4B"] = path_value
    elif model_key == "gemma3_12b":
        env["MODEL_PATH_GEMMA3_12B"] = path_value
    elif model_key == "gemma3_27b":
        env["MODEL_PATH_GEMMA3_27B"] = path_value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__, "repr": repr(value)[:200]}
    cpu = value.detach().cpu().contiguous()
    arr = cpu.numpy()
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": sha256_bytes(arr.tobytes()),
    }


def summarize_prompt_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate(items):
        copied = {"index": idx, "type": item.get("type")}
        for key in ("value", "image", "video", "text"):
            if key in item:
                value = item[key]
                if isinstance(value, str) and len(value) > 2000:
                    value = value[:2000] + f"...[TRUNCATED {len(value) - 2000} chars]"
                copied[key] = value
        if "replay_meta" in item:
            copied["replay_meta"] = item["replay_meta"]
        out.append(copied)
    return out


def count_types(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(item.get("type"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def count_signature_images(items: list[dict[str, Any]]) -> int:
    return sum(1 for item in items if str(item.get("type", "")).startswith("image"))


def processed_image_count(family: str, payload: dict[str, Any]) -> int:
    if family == "qwen25vl":
        return count_signature_images(payload.get("vllm_signature", []))
    if family == "minicpm45":
        return count_signature_images(payload.get("vllm_content_signature", []))
    if family == "gemma3":
        return len(payload.get("vllm_images", []))
    return 0


def write_probe_matrix(repo_root: Path, matrix_path: Path, result_root: Path, model_key: str, dataset: str, mode: str, policy: str) -> None:
    matrix = {
        "name": "step7_standard_entry_parity",
        "repo_root": str(repo_root),
        "results_root": str(result_root),
        "node_gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "datasets": [dataset],
        "policies": {policy: {"replay_prompt_template_name": "identity" if policy == "default" else "directly_answer"}},
        "replay_modes": [mode],
        "image_transforms": ["baseline"],
        "models": [model_key],
        "replay": {
            "replay_times": 1,
            "template_on_last_replay_text": 1,
            "image_copy_mode": "reuse_path",
            "limit_mm_per_prompt": 2,
            "safe_fallback": 0,
            "safe_truncate_chars": 6000,
            "stage_debug": 0,
            "stage_debug_samples": 0,
            "prompt_audit": 0,
            "prompt_audit_print": 0,
        },
        "answer_format": {"enable": 0},
        "evaluation": {"launch_mode": "skip", "nproc": 1, "judge": "gpt-4o-mini", "openai_api_base": "https://api.openai.com/v1"},
        "worker_monitor": {"enable": False},
        "resume_infer_default": False,
    }
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")


def runtime_env(repo_root: Path, output_root: Path, model_key: str, dataset: str, mode: str, policy: str, gpu_id: str, model_path_override: str = "", lmu_data: str = "") -> tuple[dict[str, str], Any, Any]:
    matrix_path = output_root / "_tmp" / f"matrix_{model_key}_{dataset}_{mode}_{policy}.yaml"
    write_probe_matrix(repo_root, matrix_path, output_root / "_runner_results", model_key, dataset, mode, policy)
    sys.path.insert(0, str(repo_root))
    from vlmeval.cli.run_benchmark import BenchmarkRunner

    args = SimpleNamespace(
        matrix_config=matrix_path,
        model_config=repo_root / "configs" / "models.yaml",
        nodes=1,
        node_rank=0,
        gpu_ids=gpu_id,
        models=model_key,
        policies=policy,
        modes=mode,
        transforms="baseline",
        datasets=dataset,
        task_manifest=None,
        scheduler="model_sequential",
        manifest_is_node_shard=False,
        resume_infer=False,
        plan_only=True,
    )
    runner = BenchmarkRunner(repo_root, args)
    if len(runner.tasks) != 1:
        raise RuntimeError(f"expected exactly one task, got {len(runner.tasks)}")
    task = runner.tasks[0]
    model = runner.models[model_key]
    env = runner.build_env(model, task, [gpu_id])
    if model_path_override:
        apply_model_path_override(env, model_key, model_path_override)
    if lmu_data:
        env["LMUData"] = lmu_data
    return env, runner, task


def load_qwen_model(env: dict[str, str], registry_name: str):
    with patched_environ(env):
        for name in ["vlmeval.config_runtime", "vlmeval.config_qwen_minimal"]:
            sys.modules.pop(name, None)
        cfg = importlib.import_module("vlmeval.config_runtime")
        if registry_name not in cfg.supported_VLM:
            raise KeyError(f"{registry_name} not in runtime registry: {sorted(cfg.supported_VLM)}")
        model = cfg.supported_VLM[registry_name](use_vllm=False)
        model.eval() if hasattr(model, "eval") else None
        return model


def refresh_replay_runtime(model: Any, env: dict[str, str]) -> None:
    with patched_environ(env):
        from vlmeval.vlm.replay_policy import read_replay_config_from_env
        from vlmeval.vlm.qwen2_vl.replay_prompt_template import read_prompt_template_config_from_env
        from vlmeval.vlm.replay_image_transform import canonicalize_image_transform

        model.replay_cfg = read_replay_config_from_env()
        model.prompt_template_cfg = read_prompt_template_config_from_env()
        model.template_on_last_replay_text = os.environ.get("REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT", "0").strip().lower() in {"1", "true", "yes", "on"}
        model.image_transform_name = canonicalize_image_transform(os.environ.get("REPLAY_IMAGE_TRANSFORM", "baseline"))
        model.image_transform_cache_dir = os.environ.get("REPLAY_IMAGE_TRANSFORM_CACHE_DIR", "").strip()
        model.image_transform_target_position = max(1, int(os.environ.get("REPLAY_IMAGE_TRANSFORM_TARGET_POSITION", "2")))
        model.safe_fallback_enabled = os.environ.get("REPLAY_SAFE_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}


def build_standard_prompt(model: Any, dataset_obj: Any, dataset_name: str, row: Any) -> list[dict[str, Any]]:
    from vlmeval.inference import _attach_replay_meta, _build_replay_meta, _maybe_build_prompt_struct, _normalize_resume_index

    if hasattr(model, "set_dump_image"):
        model.set_dump_image(dataset_obj.dump_image)
    index_to_position = {
        _normalize_resume_index(dataset_obj.data.iloc[pos]["index"]): pos
        for pos in range(len(dataset_obj.data))
    }
    prompt_cache: dict[int, list[dict[str, Any]]] = {}
    struct = _maybe_build_prompt_struct(model, dataset_obj, dataset_name, row)
    replay_meta = _build_replay_meta(
        model=model,
        dataset=dataset_obj,
        dataset_name=dataset_name,
        current_row=row,
        current_struct=struct,
        index_to_position=index_to_position,
        prompt_cache=prompt_cache,
    )
    return _attach_replay_meta(struct, replay_meta)


def topk_json(logits: np.ndarray, tokenizer: Any, k: int = 256) -> list[dict[str, Any]]:
    k = min(k, logits.shape[-1])
    idx = np.argpartition(-logits, k - 1)[:k]
    idx = idx[np.argsort(-logits[idx])]
    out = []
    for rank, token_id in enumerate(idx.tolist()):
        out.append({
            "rank": rank,
            "token_id": int(token_id),
            "logit": float(logits[token_id]),
            "token": tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False),
        })
    return out


def forward_artifact(model: Any, prompt: list[dict[str, Any]], dataset_name: str, out_dir: Path) -> dict[str, Any]:
    try:
        from qwen_vl_utils import process_vision_info
    except Exception as err:
        raise RuntimeError("qwen_vl_utils is required for Qwen2.5-VL parity probing") from err

    messages = []
    if getattr(model, "system_prompt", None) is not None:
        messages.append({"role": "system", "content": model.system_prompt})
    hf_content = model._prepare_content(prompt, dataset=dataset_name)
    vllm_content = model._prepare_content_vllm(prompt, dataset=dataset_name)
    messages.append({"role": "user", "content": hf_content})
    text = model.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
    text_for_hash = json.dumps(text, ensure_ascii=False, sort_keys=True) if isinstance(text, list) else str(text)
    images, videos = process_vision_info([messages])
    inputs = model.processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")
    input_summaries = {key: tensor_summary(value) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
    inputs = inputs.to("cuda")
    with torch.inference_mode():
        outputs = model.model(**inputs, use_cache=False)
    logits = outputs.logits[:, -1, :].detach().float().cpu().numpy()[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "next_token_logits.npy", logits)
    token_ids = inputs.input_ids[0].detach().cpu().tolist()
    tokenizer = model.processor.tokenizer
    top = topk_json(logits, tokenizer, k=256)
    write_json(out_dir / "top256.json", top)
    return {
        "messages": summarize_prompt_items(messages),
        "hf_content": summarize_prompt_items(hf_content),
        "vllm_content": summarize_prompt_items(vllm_content),
        "hf_counts": count_types(hf_content),
        "vllm_counts": count_types(vllm_content),
        "chat_template_text": text,
        "chat_template_sha256": sha256_bytes(text_for_hash.encode("utf-8")),
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "input_tensors": input_summaries,
        "top1": top[0],
        "top256_path": str(out_dir / "top256.json"),
        "logits_path": str(out_dir / "next_token_logits.npy"),
    }


def env_subset(env: dict[str, str]) -> dict[str, str]:
    keys = [
        "MODEL_PATH", "MODEL_PATH_QWEN25_3B", "MODEL_PATH_QWEN25", "MODEL_PATH_QWEN25_32B", "MODEL_PATH_QWEN25_72B",
        "MODEL_PATH_MINICPM45", "MODEL_PATH_MINICPMO45", "MODEL_PATH_GEMMA3_4B", "MODEL_PATH_GEMMA3_12B", "MODEL_PATH_GEMMA3_27B",
        "CUDA_VISIBLE_DEVICES", "REPLAY_MODE", "REPLAY_PROMPT_TEMPLATE_NAME", "REPLAY_TIMES", "REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT",
        "REPLAY_SAFE_FALLBACK", "REPLAY_LIMIT_MM_PER_PROMPT", "DYNAMATH_PROMPT_SCHEMA", "VLLM_USE_V1", "VLLM_MAX_NUM_SEQS", "VLLM_TP_SIZE", "VLLM_MAX_MODEL_LEN",
        "LOGICVISTA_QWEN25VL_LEGACY_SAMPLING", "LOGICVISTA_QWEN25VL_BATCH_SIZE", "LOGICVISTA_QWEN25VL_MAX_NUM_SEQS",
        "QWEN2VL_VLLM_REPETITION_PENALTY", "QWEN2VL_VLLM_TEMPERATURE", "QWEN2VL_VLLM_TOP_P",
        "QWEN2VL_VLLM_TOP_K", "QWEN2VL_VLLM_MAX_TOKENS", "QWEN2VL_VLLM_STOP_TOKEN_IDS",
        "MINICPM45_USE_VLLM", "MINICPM45_VLLM_TP_SIZE", "MINICPM45_VLLM_MAX_NUM_SEQS", "MINICPM45_VLLM_MAX_MODEL_LEN", "MINICPM45_MAX_NEW_TOKENS", "MINICPM45_REASONING_MODE",
        "GEMMA3_USE_VLLM", "GEMMA3_VLLM_TP_SIZE", "GEMMA3_VLLM_MAX_NUM_SEQS", "GEMMA3_VLLM_MAX_MODEL_LEN", "GEMMA3_MAX_NEW_TOKENS", "GEMMA3_VLLM_TEMPERATURE", "GEMMA3_TEMPERATURE",
        "VLMEVAL_API_MAX_TOKENS", "VLMEVAL_API_TIMEOUT", "VLMEVAL_API_IMG_SIZE",
    ]
    return {key: env[key] for key in keys if key in env}


@REGISTRY.register("qwen25vl_prefill")
def dump_qwen25vl_prefill(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    sys.path.insert(0, str(repo_root))
    from vlmeval.dataset import build_dataset

    first_env = None
    first_registry = None
    cases = []
    for dataset_name in args.datasets:
        for mode_label in args.modes:
            mode = REPLAY_MODES.get(mode_label, mode_label)
            env, runner, task = runtime_env(
                repo_root,
                output_root,
                args.model_key,
                dataset_name,
                mode,
                args.policy,
                args.gpu_id,
                model_path_override=args.model_path_override,
                lmu_data=args.lmu_data,
            )
            model_spec = runner.models[args.model_key]
            if first_env is None:
                first_env = env
                first_registry = model_spec.registry_name
            cases.append((dataset_name, mode_label, mode, env, model_spec.registry_name))
    if first_env is None or first_registry is None:
        raise RuntimeError("no cases selected")

    model = load_qwen_model(first_env, first_registry)
    for dataset_name, mode_label, mode, env, registry_name in cases:
        if registry_name != first_registry:
            raise RuntimeError("single-process probe expects one registry")
        refresh_replay_runtime(model, env)
        with patched_environ(env):
            dataset_obj = build_dataset(dataset_name)
            row = dataset_obj.data.iloc[int(args.row)]
            prompt = build_standard_prompt(model, dataset_obj, dataset_name, row)
            case_dir = output_root / dataset_name / mode_label
            artifact = forward_artifact(model, prompt, dataset_name, case_dir)
            prompt_counts = count_types(prompt)
            expected_images = EXPECTED_IMAGE_COUNTS[mode]
            checks = {
                "expected_image_count": expected_images,
                "prompt_image_count": prompt_counts.get("image", 0),
                "hf_image_count": artifact["hf_counts"].get("image", 0),
                "vllm_image_count": artifact["vllm_counts"].get("image", 0),
                "image_count_ok": artifact["hf_counts"].get("image", 0) == expected_images,
                "dynamath_legacy_prompt_ok": None,
                "logicvista_qwen_sampling_ok": None,
            }
            prompt_text = "\n".join(str(x.get("value") or x.get("text") or "") for x in prompt if x.get("type") == "text")
            if dataset_name == "DynaMath":
                checks["dynamath_legacy_prompt_ok"] = ("solution" in prompt_text and "short answer" in prompt_text)
            if dataset_name == "LogicVista":
                checks["logicvista_qwen_sampling_ok"] = all(env.get(k) == v for k, v in LOGICVISTA_QWEN_SAMPLING.items())
            payload = {
                "repo_root": str(repo_root),
                "model_key": args.model_key,
                "registry_name": registry_name,
                "dataset": dataset_name,
                "row_position": int(args.row),
                "sample_index": json_default(row.get("index")),
                "mode_label": mode_label,
                "replay_mode": mode,
                "policy": args.policy,
                "runtime_env": env_subset(env),
                "dataset_columns": list(dataset_obj.data.columns),
                "standard_prompt": summarize_prompt_items(prompt),
                "standard_prompt_counts": prompt_counts,
                "checks": checks,
                "forward": artifact,
            }
            write_json(case_dir / "artifact.json", payload)
            print(json.dumps({"case": f"{dataset_name}/{mode_label}", "artifact": str(case_dir / "artifact.json"), "checks": checks}, ensure_ascii=False), flush=True)


def compare_case(release_case: Path, main_case: Path) -> dict[str, Any]:
    rel = json.loads((release_case / "artifact.json").read_text(encoding="utf-8"))
    main = json.loads((main_case / "artifact.json").read_text(encoding="utf-8"))
    rel_logits = np.load(release_case / "next_token_logits.npy")
    main_logits = np.load(main_case / "next_token_logits.npy")
    diff = np.abs(rel_logits - main_logits)
    token_ids_equal = rel["forward"]["token_ids"] == main["forward"]["token_ids"]
    chat_equal = rel["forward"]["chat_template_text"] == main["forward"]["chat_template_text"]
    top1_equal = rel["forward"]["top1"]["token_id"] == main["forward"]["top1"]["token_id"]
    return {
        "dataset": rel["dataset"],
        "mode_label": rel["mode_label"],
        "token_ids_equal": token_ids_equal,
        "chat_template_equal": chat_equal,
        "input_tensor_hashes_equal": rel["forward"].get("input_tensors") == main["forward"].get("input_tensors"),
        "top1_equal": top1_equal,
        "release_top1": rel["forward"]["top1"],
        "main_top1": main["forward"]["top1"],
        "max_abs_logit_diff": float(diff.max()),
        "mean_abs_logit_diff": float(diff.mean()),
        "release_checks": rel["checks"],
        "main_checks": main["checks"],
        "pass": token_ids_equal and chat_equal and top1_equal and float(diff.max()) <= 1e-3 and all(v is not False for v in rel["checks"].values()) and all(v is not False for v in main["checks"].values()),
    }


@REGISTRY.register("compare")
def compare_outputs(args: argparse.Namespace) -> None:
    release_root = Path(args.release_root).resolve()
    main_root = Path(args.main_root).resolve()
    output = Path(args.output).resolve()
    rows = []
    for dataset in args.datasets:
        for mode_label in args.modes:
            rows.append(compare_case(release_root / dataset / mode_label, main_root / dataset / mode_label))
    summary = {"rows": rows, "all_pass": all(row["pass"] for row in rows)}
    write_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["all_pass"]:
        raise SystemExit(2)


def purge_target_imports() -> None:
    for name in list(sys.modules):
        if name == "vlmeval" or name.startswith("vlmeval."):
            sys.modules.pop(name, None)


def repo_git_snapshot(repo_root: Path) -> dict[str, Any]:
    import subprocess
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=repo_root, stderr=subprocess.STDOUT, text=True).strip()
        except Exception as err:
            return f"ERROR: {type(err).__name__}: {err}"
    return {
        "repo_root": str(repo_root),
        "head": run(["git", "rev-parse", "--short", "HEAD"]),
        "branch": run(["git", "branch", "--show-current"]),
        "status_short": run(["git", "status", "--short"]),
    }


def file_manifest(path: str) -> dict[str, Any]:
    p = Path(path)
    required_any = ["preprocessor_config.json", "processor_config.json"]
    required = ["config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json"]
    files = {name: (p / name).exists() for name in required}
    files["processor_or_preprocessor"] = any((p / name).exists() for name in required_any)
    index = p / "model.safetensors.index.json"
    shards = sorted(p.glob("*.safetensors"))[:3]
    return {
        "path": str(p),
        "exists": p.exists(),
        "is_symlink": p.is_symlink(),
        "realpath": str(p.resolve()) if p.exists() else None,
        "required_files": files,
        "has_model_index": index.exists(),
        "sample_shards": [s.name for s in shards],
        "shard_count": len(list(p.glob("*.safetensors"))),
        "ok": p.exists() and all(files.values()) and index.exists() and bool(shards),
    }


def image_info(path_value: str) -> dict[str, Any]:
    raw = str(path_value)
    path = raw[7:] if raw.startswith("file://") else raw
    info = {"path": raw, "exists": Path(path).exists()}
    if not info["exists"]:
        return info
    try:
        from PIL import Image
        with Image.open(path) as img:
            info.update({"size": list(img.size), "mode": img.mode})
    except Exception as err:
        info["image_error"] = f"{type(err).__name__}: {err}"
    try:
        info["sha256"] = sha256_bytes(Path(path).read_bytes())
    except Exception as err:
        info["sha256_error"] = f"{type(err).__name__}: {err}"
    return info


def item_signature(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sig = []
    for item in items:
        typ = item.get("type")
        row = {"type": typ}
        text = item.get("text", item.get("value"))
        if typ == "text":
            row["text"] = str(text)
            row["text_sha256"] = sha256_bytes(str(text).encode("utf-8"))
        elif typ == "image":
            value = item.get("value") or item.get("image") or item.get("url") or item.get("path")
            row["image"] = image_info(str(value))
        elif typ == "image_pil":
            img = item.get("image_pil")
            row["image_pil"] = {"size": list(getattr(img, "size", [])), "mode": getattr(img, "mode", None)}
        else:
            row["repr"] = repr(item)[:500]
        sig.append(row)
    return sig


def summarize_sampling_params(obj: Any) -> dict[str, Any]:
    out = {"type": type(obj).__name__, "repr": repr(obj)}
    for key in ["max_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "presence_penalty"]:
        if hasattr(obj, key):
            out[key] = json_default(getattr(obj, key))
    return out


def make_infer_command(runner: Any, task: Any, model: Any, env: dict[str, str]) -> list[str]:
    cmd = [
        runner.env_profiles[model.env_profile].python,
        "run.py",
        "--data", task.dataset,
        "--model", model.registry_name,
        "--work-dir", str(runner.task_root(task)),
        "--mode", "infer",
        "--verbose",
        "--batch-size", str(runner.infer_batch_size_for_task(model, task)),
    ]
    if hasattr(runner, "prediction_dir"):
        cmd.extend(["--pred-output-dir", str(runner.prediction_dir(task)), "--no-link-predictions"])
    return cmd


def context_checks(model_key: str, dataset: str, env: dict[str, str], model_spec: Any, runner: Any, task: Any) -> dict[str, Any]:
    family = model_family(model_key)
    checks: dict[str, Any] = {
        "family": family,
        "safe_fallback_fail_closed": env.get("REPLAY_SAFE_FALLBACK") == "0",
        "strict_batch": None if "VLMEVAL_STRICT_BATCH" not in env else env.get("VLMEVAL_STRICT_BATCH") == "1",
        "dynamath_prompt_schema": None,
        "logicvista_qwen_v0_policy": None,
        "non_qwen_no_logicvista_v0_override": None,
        "api_profile_ok": None,
        "api_minimal_config_ok": None,
        "api_no_open_source_minimal_config": None,
        "standard_route_backend": "api" if family == "api" else "vllm",
        "logits_probe_backend": None,
    }
    if dataset == "DynaMath":
        expected_schema = "legacy_two_keys" if family == "qwen25vl" else "short_answer_only"
        checks["dynamath_prompt_schema"] = env.get("DYNAMATH_PROMPT_SCHEMA") == expected_schema
    if dataset == "LogicVista" and family == "qwen25vl":
        checks["logicvista_qwen_v0_policy"] = env.get("VLLM_USE_V1") == "0" and all(env.get(k) == v for k, v in LOGICVISTA_QWEN_SAMPLING.items())
    if dataset == "LogicVista" and family in {"minicpm45", "gemma3"}:
        checks["non_qwen_no_logicvista_v0_override"] = env.get("VLLM_USE_V1", "1") != "0"
    if family == "api":
        checks["api_profile_ok"] = model_spec.env_profile == "api_replay"
        checks["api_minimal_config_ok"] = env.get("VLMEVAL_USE_API_REPLAY_MINIMAL_CONFIG") == "1"
        checks["api_no_open_source_minimal_config"] = not any(
            env.get(key)
            for key in (
                "VLMEVAL_USE_QWEN_MINIMAL_CONFIG",
                "VLMEVAL_USE_MINICPM45_MINIMAL_CONFIG",
                "VLMEVAL_USE_GEMMA3_MINIMAL_CONFIG",
            )
        )
    return checks


@REGISTRY.register("preflight")
def dump_preflight(args: argparse.Namespace) -> None:
    repo_roots = [Path(p).resolve() for p in args.repo_roots]
    payload: dict[str, Any] = {
        "host": os.uname().nodename,
        "python": sys.version,
        "repos": [repo_git_snapshot(p) for p in repo_roots],
        "model_root": args.model_root,
        "model_cp_root": args.model_cp_root,
        "models": {},
        "datasets": {},
        "versions": {},
        "output_contract": {
            "thresholds": {"max_abs_logit_diff": 1e-3},
            "standard_route_backend_note": "Open-source standard routes are vLLM; HF logits are only same-prefix probes.",
        },
    }
    for key in ACTIVE_MODEL_KEYS:
        payload["models"][key] = file_manifest(resolved_model_path(key, args.model_root, args.model_cp_root))
    try:
        import torch, transformers, vllm
        payload["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__, "vllm": getattr(vllm, "__version__", "unknown")}
    except Exception as err:
        payload["versions_error"] = f"{type(err).__name__}: {err}"
    # Use the first repo for real dataset row evidence.
    repo_root = repo_roots[0]
    sys.path.insert(0, str(repo_root))
    purge_target_imports()
    sys.path.insert(0, str(repo_root))
    with patched_environ({"LMUData": args.lmu_data, "MODEL_ROOT": args.model_root}):
        from vlmeval.dataset import build_dataset
        for dataset_name in args.datasets:
            ds = build_dataset(dataset_name)
            row = ds.data.iloc[int(args.row)]
            prompt = ds.build_prompt(row)
            payload["datasets"][dataset_name] = {
                "rows": len(ds.data),
                "row_position": int(args.row),
                "sample_index": json_default(row.get("index")),
                "columns": list(ds.data.columns),
                "question_head": str(row.get("question", ""))[:1000],
                "prompt_signature": item_signature(prompt),
            }
    write_json(Path(args.output), payload)
    print(json.dumps({"preflight": args.output}, ensure_ascii=False), flush=True)


@REGISTRY.register("runner_context_dump")
def dump_runner_context(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    models = split_csv(" ".join(args.models)) if args.models else ACTIVE_MODEL_KEYS
    rows = []
    for model_key in models:
        override = resolved_model_path(model_key, args.model_root, args.model_cp_root)
        for dataset in args.datasets:
            for mode_label in args.modes:
                mode = REPLAY_MODES.get(mode_label, mode_label)
                env, runner, task = runtime_env(repo_root, output_root, model_key, dataset, mode, args.policy, args.gpu_id, model_path_override=override, lmu_data=args.lmu_data)
                model_spec = runner.models[model_key]
                row = {
                    "repo_root": str(repo_root),
                    "repo_snapshot": repo_git_snapshot(repo_root),
                    "model_key": model_key,
                    "family": model_family(model_key),
                    "registry_name": model_spec.registry_name,
                    "dataset": dataset,
                    "mode_label": mode_label,
                    "replay_mode": mode,
                    "policy": args.policy,
                    "model_path_config": str(model_spec.model_path),
                    "model_path_resolved": override,
                    "model_spec": {k: json_default(v) for k, v in vars(model_spec).items()},
                    "infer_batch_size": runner.infer_batch_size_for_task(model_spec, task),
                    "max_num_seqs": runner.max_num_seqs_for_task(model_spec, task),
                    "env": env_subset(env),
                    "infer_command": make_infer_command(runner, task, model_spec, env),
                    "checks": context_checks(model_key, dataset, env, model_spec, runner, task),
                }
                case_dir = output_root / model_key / dataset / mode_label
                write_json(case_dir / "context.json", row)
                rows.append(row)
    summary = {"repo_root": str(repo_root), "rows": rows, "all_context_checks_pass": all(all(v is not False for v in r["checks"].values()) for r in rows)}
    write_json(output_root / "context_summary.json", summary)
    print(json.dumps({"context_summary": str(output_root / "context_summary.json"), "rows": len(rows), "all_pass": summary["all_context_checks_pass"]}, ensure_ascii=False), flush=True)
    if not summary["all_context_checks_pass"]:
        raise SystemExit(2)


def patch_heavy_model_loads(family: str):
    stack = contextlib.ExitStack()
    class DummyModel:
        device = "cuda"
        def to(self, *args, **kwargs): return self
        def eval(self): return self
        def cuda(self): return self
        def chat(self, *args, **kwargs): return ""
        def generate(self, *args, **kwargs): raise RuntimeError("DummyModel cannot generate")
        def __call__(self, *args, **kwargs): raise RuntimeError("DummyModel cannot forward")
    if family == "qwen25vl":
        import transformers
        for name in ["Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration"]:
            cls = getattr(transformers, name, None)
            if cls is not None:
                stack.enter_context(mock.patch.object(cls, "from_pretrained", classmethod(lambda cls, *a, **k: DummyModel())))
    elif family == "minicpm45":
        import vlmeval.vlm.minicpm_v_4_5_replay as mod
        stack.enter_context(mock.patch.object(mod.AutoModel, "from_pretrained", lambda *a, **k: DummyModel()))
    elif family == "gemma3":
        import vlmeval.vlm.gemma3_replay as mod
        stack.enter_context(mock.patch.object(mod, "_load_gemma3_transformers_model", lambda *a, **k: DummyModel()))
    return stack


def load_light_model(env: dict[str, str], registry_name: str, family: str):
    with patched_environ(env):
        for name in ["vlmeval.config_runtime", "vlmeval.config_qwen_minimal", "vlmeval.config_minicpm45_minimal", "vlmeval.config_gemma3_minimal"]:
            sys.modules.pop(name, None)
        cfg = importlib.import_module("vlmeval.config_runtime")
        if registry_name not in cfg.supported_VLM:
            raise KeyError(f"{registry_name} not in runtime registry: {sorted(cfg.supported_VLM)}")
        with patch_heavy_model_loads(family):
            model = cfg.supported_VLM[registry_name](use_vllm=False)
        return model


def payload_for_model(model: Any, family: str, prompt: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    if family == "qwen25vl":
        hf_content = model._prepare_content(prompt, dataset=dataset)
        vllm_content = model._prepare_content_vllm(prompt, dataset=dataset)
        text = None
        token_ids = None
        try:
            messages = []
            if getattr(model, "system_prompt", None) is not None:
                messages.append({"role": "system", "content": model.system_prompt})
            messages.append({"role": "user", "content": hf_content})
            text = model.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
            tokenized = model.processor.tokenizer(text, add_special_tokens=False)
            token_ids = tokenized.get("input_ids")
        except Exception as err:
            text = f"CHAT_TEMPLATE_ERROR: {type(err).__name__}: {err}"
        return {"hf_content": summarize_prompt_items(hf_content), "vllm_content": summarize_prompt_items(vllm_content), "hf_signature": item_signature(hf_content), "vllm_signature": item_signature(vllm_content), "chat_template_text": text, "token_ids": token_ids, "logits_probe_backend": None}
    if family == "minicpm45":
        replayed = model._apply_replay_pipeline(prompt, dataset=dataset)
        vllm_content = model._message_to_vllm_content(replayed, dataset=dataset)
        sampling, chat_template_kwargs = model._build_vllm_sampling(dataset=dataset)
        return {
            "replayed": summarize_prompt_items(replayed),
            "replayed_signature": item_signature(replayed),
            "vllm_content_signature": item_signature(vllm_content),
            "sampling_params": summarize_sampling_params(sampling),
            "chat_template_kwargs": chat_template_kwargs,
            "reasoning_mode": model._reasoning_mode_override(),
            "use_upsize": model.use_upsize(dataset),
            "logits_probe_backend": None,
        }
    if family == "gemma3":
        replayed = model._apply_replay_pipeline(prompt, dataset=dataset)
        hf_messages = model._message_to_hf_messages(replayed)
        vllm_payload = model._message_to_vllm_payload(replayed)
        sampling = model._build_sampling_params()
        token_ids = None
        try:
            tokenized = model.processor(text=vllm_payload["prompt"], return_tensors=None)
            token_ids = tokenized.get("input_ids")
        except Exception:
            token_ids = None
        return {
            "replayed": summarize_prompt_items(replayed),
            "replayed_signature": item_signature(replayed),
            "hf_messages": hf_messages,
            "vllm_prompt": vllm_payload.get("prompt"),
            "vllm_images": [image_info(getattr(img, "filename", "")) | {"size": list(getattr(img, "size", [])), "mode": getattr(img, "mode", None)} for img in vllm_payload.get("multi_modal_data", {}).get("image", [])],
            "sampling_params": summarize_sampling_params(sampling),
            "token_ids": token_ids,
            "logits_probe_backend": None,
        }
    raise ValueError(f"unsupported payload family: {family}")


@REGISTRY.register("standard_payload_dump")
def dump_standard_payload(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    sys.path.insert(0, str(repo_root))
    from vlmeval.dataset import build_dataset
    models = split_csv(" ".join(args.models)) if args.models else ACTIVE_MODEL_KEYS
    rows = []
    dataset_cache = {}
    for model_key in models:
        family = model_family(model_key)
        if family == "api":
            continue
        override = resolved_model_path(model_key, args.model_root, args.model_cp_root)
        first_model = None
        first_registry = None
        for dataset_name in args.datasets:
            for mode_label in args.modes:
                mode = REPLAY_MODES.get(mode_label, mode_label)
                env, runner, task = runtime_env(repo_root, output_root, model_key, dataset_name, mode, args.policy, args.gpu_id, model_path_override=override, lmu_data=args.lmu_data)
                registry_name = runner.models[model_key].registry_name
                if first_model is None or first_registry != registry_name:
                    first_model = load_light_model(env, registry_name, family)
                    first_registry = registry_name
                model = first_model
                if family == "qwen25vl":
                    refresh_replay_runtime(model, env)
                elif family in {"minicpm45", "gemma3"}:
                    # Re-read replay/template env for the reused lightweight instance.
                    with patched_environ(env):
                        from vlmeval.vlm.replay_policy import read_replay_config_from_env
                        from vlmeval.vlm.qwen2_vl.replay_prompt_template import read_prompt_template_config_from_env
                        model.replay_cfg = read_replay_config_from_env()
                        model.prompt_template_cfg = read_prompt_template_config_from_env()
                        model.template_on_last_replay_text = os.environ.get("REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT", "0").strip().lower() in {"1", "true", "yes", "on"}
                with patched_environ(env):
                    if dataset_name not in dataset_cache:
                        dataset_cache[dataset_name] = build_dataset(dataset_name)
                    dataset_obj = dataset_cache[dataset_name]
                    row = dataset_obj.data.iloc[int(args.row)]
                    prompt = build_standard_prompt(model, dataset_obj, dataset_name, row)
                    payload = payload_for_model(model, family, prompt, dataset_name)
                prompt_counts = count_types(prompt)
                expected_images = EXPECTED_IMAGE_COUNTS[mode]
                processed_images = processed_image_count(family, payload)
                checks = {
                    "expected_image_count": expected_images,
                    "raw_prompt_image_count": prompt_counts.get("image", 0),
                    "processed_image_count": processed_images,
                    "processed_image_count_ok": processed_images == expected_images,
                    "dynamath_qwen_prompt_policy_ok": None,
                    "dynamath_non_qwen_prompt_policy_ok": None,
                    "logicvista_qwen_sampling_ok": None,
                    "minicpm_no_upsize_ok": None,
                    "gemma3_temperature_ok": None,
                }
                text_blob = "\n".join(str(x.get("value") or x.get("text") or "") for x in prompt if x.get("type") == "text")
                if dataset_name == "DynaMath":
                    lowered_prompt = text_blob.lower()
                    has_solution = "solution" in lowered_prompt
                    has_short_answer = "short answer" in lowered_prompt
                    has_direct_answer = env.get("REPLAY_PROMPT_TEMPLATE_NAME") == "directly_answer"
                    has_two_key_legacy = "two keys" in lowered_prompt and "solution" in lowered_prompt
                    if family == "qwen25vl":
                        if args.policy == "default":
                            checks["dynamath_qwen_prompt_policy_ok"] = has_solution and has_short_answer
                        else:
                            checks["dynamath_qwen_prompt_policy_ok"] = (not has_solution) and has_direct_answer
                    elif family in {"minicpm45", "gemma3"}:
                        if args.policy == "default":
                            checks["dynamath_non_qwen_prompt_policy_ok"] = (not has_solution) and (not has_two_key_legacy)
                        else:
                            checks["dynamath_non_qwen_prompt_policy_ok"] = (not has_solution) and has_direct_answer
                if dataset_name == "LogicVista" and family == "qwen25vl":
                    checks["logicvista_qwen_sampling_ok"] = all(env.get(k) == v for k, v in LOGICVISTA_QWEN_SAMPLING.items())
                if family == "minicpm45":
                    checks["minicpm_no_upsize_ok"] = payload.get("use_upsize") is False
                if family == "gemma3":
                    checks["gemma3_temperature_ok"] = float(env.get("GEMMA3_VLLM_TEMPERATURE", env.get("GEMMA3_TEMPERATURE", "0") or "0")) == 0.0
                case = {
                    "repo_root": str(repo_root),
                    "repo_snapshot": repo_git_snapshot(repo_root),
                    "model_key": model_key,
                    "family": family,
                    "registry_name": registry_name,
                    "dataset": dataset_name,
                    "row_position": int(args.row),
                    "sample_index": json_default(row.get("index")),
                    "mode_label": mode_label,
                    "replay_mode": mode,
                    "policy": args.policy,
                    "runtime_env": env_subset(env),
                    "standard_route_backend": "vllm",
                    "standard_prompt": summarize_prompt_items(prompt),
                    "standard_prompt_signature": item_signature(prompt),
                    "standard_prompt_counts": prompt_counts,
                    "payload": payload,
                    "checks": checks,
                }
                case_dir = output_root / model_key / dataset_name / mode_label
                write_json(case_dir / "payload.json", case)
                rows.append(case)
                print(json.dumps({"payload": str(case_dir / "payload.json"), "checks": checks}, ensure_ascii=False), flush=True)
    summary = {"rows": rows, "all_payload_checks_pass": all(all(v is not False for v in r["checks"].values()) for r in rows)}
    write_json(output_root / "payload_summary.json", summary)
    if not summary["all_payload_checks_pass"]:
        raise SystemExit(2)


def comparable_payload(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_key": case["model_key"],
        "family": case["family"],
        "dataset": case["dataset"],
        "mode_label": case["mode_label"],
        "replay_mode": case["replay_mode"],
        "standard_prompt_signature": case.get("standard_prompt_signature"),
        "payload": case.get("payload"),
        "checks": case.get("checks"),
    }


@REGISTRY.register("compare_payloads")
def compare_payloads(args: argparse.Namespace) -> None:
    release_root = Path(args.release_root).resolve()
    main_root = Path(args.main_root).resolve()
    rows = []
    models = split_csv(" ".join(args.models))
    for model_key in models:
        for dataset in args.datasets:
            for mode_label in args.modes:
                rel_path = release_root / model_key / dataset / mode_label / "payload.json"
                main_path = main_root / model_key / dataset / mode_label / "payload.json"
                rel = json.loads(rel_path.read_text())
                main = json.loads(main_path.read_text())
                rel_cmp = comparable_payload(rel)
                main_cmp = comparable_payload(main)
                equal = rel_cmp == main_cmp
                rows.append({"model_key": model_key, "dataset": dataset, "mode_label": mode_label, "equal": equal, "release_path": str(rel_path), "main_path": str(main_path)})
    summary = {"rows": rows, "all_pass": all(r["equal"] for r in rows)}
    write_json(Path(args.output), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["all_pass"]:
        raise SystemExit(2)




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standard-entry prompt/logit parity probes.")
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--repo-roots", nargs="+", required=True)
    pre.add_argument("--output", required=True)
    pre.add_argument("--datasets", nargs="+", default=["DynaMath", "LogicVista"])
    pre.add_argument("--row", type=int, default=0)
    pre.add_argument("--model-root", default="/user/zyc1781/models")
    pre.add_argument("--model-cp-root", default="/user/zyc1781/models-cp")
    pre.add_argument("--lmu-data", default="/user/zyc1781/LMUData")
    pre.add_argument("--probe", default="preflight")

    ctx = sub.add_parser("context")
    ctx.add_argument("--repo-root", required=True)
    ctx.add_argument("--output-root", required=True)
    ctx.add_argument("--models", nargs="+", default=ACTIVE_MODEL_KEYS)
    ctx.add_argument("--policy", default="default", choices=["default", "direct"])
    ctx.add_argument("--datasets", nargs="+", default=["DynaMath", "LogicVista"])
    ctx.add_argument("--modes", nargs="+", default=list(REPLAY_MODES.keys()))
    ctx.add_argument("--gpu-id", default="0")
    ctx.add_argument("--model-root", default="/user/zyc1781/models")
    ctx.add_argument("--model-cp-root", default="/user/zyc1781/models-cp")
    ctx.add_argument("--lmu-data", default="/user/zyc1781/LMUData")
    ctx.add_argument("--probe", default="runner_context_dump")

    payload = sub.add_parser("payload")
    payload.add_argument("--repo-root", required=True)
    payload.add_argument("--output-root", required=True)
    payload.add_argument("--models", nargs="+", default=ACTIVE_MODEL_KEYS)
    payload.add_argument("--policy", default="default", choices=["default", "direct"])
    payload.add_argument("--datasets", nargs="+", default=["DynaMath", "LogicVista"])
    payload.add_argument("--modes", nargs="+", default=list(REPLAY_MODES.keys()))
    payload.add_argument("--row", type=int, default=0)
    payload.add_argument("--gpu-id", default="0")
    payload.add_argument("--model-root", default="/user/zyc1781/models")
    payload.add_argument("--model-cp-root", default="/user/zyc1781/models-cp")
    payload.add_argument("--lmu-data", default="/user/zyc1781/LMUData")
    payload.add_argument("--probe", default="standard_payload_dump")

    dump = sub.add_parser("dump")
    dump.add_argument("--repo-root", required=True)
    dump.add_argument("--output-root", required=True)
    dump.add_argument("--model-key", default="qwen25vl_3b")
    dump.add_argument("--policy", default="default", choices=["default", "direct"])
    dump.add_argument("--datasets", nargs="+", default=["DynaMath", "LogicVista"])
    dump.add_argument("--modes", nargs="+", default=list(REPLAY_MODES.keys()))
    dump.add_argument("--row", type=int, default=0)
    dump.add_argument("--gpu-id", default="0")
    dump.add_argument("--model-path-override", default="")
    dump.add_argument("--lmu-data", default="")
    dump.add_argument("--probe", default="qwen25vl_prefill")

    cmp_payload = sub.add_parser("compare-payload")
    cmp_payload.add_argument("--release-root", required=True)
    cmp_payload.add_argument("--main-root", required=True)
    cmp_payload.add_argument("--output", required=True)
    cmp_payload.add_argument("--models", nargs="+", default=ACTIVE_MODEL_KEYS)
    cmp_payload.add_argument("--datasets", nargs="+", default=["DynaMath", "LogicVista"])
    cmp_payload.add_argument("--modes", nargs="+", default=list(REPLAY_MODES.keys()))
    cmp_payload.add_argument("--probe", default="compare_payloads")

    cmp_parser = sub.add_parser("compare")
    cmp_parser.add_argument("--release-root", required=True)
    cmp_parser.add_argument("--main-root", required=True)
    cmp_parser.add_argument("--output", required=True)
    cmp_parser.add_argument("--datasets", nargs="+", default=["DynaMath", "LogicVista"])
    cmp_parser.add_argument("--modes", nargs="+", default=list(REPLAY_MODES.keys()))
    cmp_parser.add_argument("--probe", default="compare")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    REGISTRY.get(args.probe)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
