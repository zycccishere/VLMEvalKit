from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vlmeval.probes.standard_entry_parity import (
    build_standard_prompt,
    env_subset,
    load_qwen_model,
    patched_environ,
    refresh_replay_runtime,
    repo_git_snapshot,
    runtime_env,
    summarize_prompt_items,
)


DATASET_TARGETS = {
    "DynaMath": {
        "choice_count": 4,
        "subset": "four_choice",
        "all_subset": "all_single_choice",
    },
    "WeMath": {
        "choice_count": 5,
        "subset": "five_choice",
        "all_subset": "all_single_choice",
    },
    "MMBench_DEV_EN_V11": {
        "choice_count": 4,
        "subset": "canonical_four_choice",
    },
    "SEEDBench2_Plus": {
        "choice_count": 4,
        "subset": "four_choice",
        "all_subset": "all_single_choice",
    },
}
DEFAULT_FIXED_DATASETS = ("DynaMath", "WeMath", "MMBench_DEV_EN_V11")
ALL_SINGLE_CHOICE_DATASETS = ("DynaMath", "WeMath", "SEEDBench2_Plus")
SELECTION_PROFILES = ("fixed_choice", "all_single_choice")
EXPECTED_ALL_SINGLE_CHOICE_COUNTS = {
    "DynaMath": {"rows": 1736, "choice_counts": {2: 600, 3: 420, 4: 530, 5: 166, 6: 20}},
    "WeMath": {"rows": 1732, "choice_counts": {4: 559, 5: 1168, 6: 3, 7: 2}},
    "SEEDBench2_Plus": {"rows": 2274, "choice_counts": {3: 3, 4: 2271}},
}
EXPECTED_REUSE_COUNTS = {"DynaMath": 530, "WeMath": 1168}
EXPECTED_ALL_SINGLE_CHOICE_TOTAL = 5742
EXPECTED_MISSING_SINGLE_CHOICE_TOTAL = 4044
CONDITIONS = ("baseline", "readout_v2", "full")
REPLAY_MODE = "image_text_image"
POLICY = "direct"
ANSWER_PREFIX = "Answer: "
DEFAULT_MATRIX_CONFIG = "configs/matrix.yaml"
BATCH_SINGLE_CANDIDATE_ATOL = 0.25
SCORING_CONTRACT_GLOBALS = {
    "CONDITIONS",
    "REPLAY_MODE",
    "POLICY",
    "ANSWER_PREFIX",
    "BATCH_SINGLE_CANDIDATE_ATOL",
}
SCORING_CONTRACT_FUNCTIONS = {
    "build_runtime",
    "token_id",
    "contiguous_token_spans",
    "qwen_image_spans",
    "longest_common_prefix",
    "candidate_token_plan",
    "extend_sequence_inputs",
    "derive_prefill_boundary",
    "prepare_inputs",
    "allowed_masks",
    "additive_mask",
    "repeat_qwen_inputs",
    "install_qwen_position_ids",
    "candidate_scores",
    "score_from_values",
    "clone_inputs",
    "corrupt_blocked_question_tokens",
    "model_dtype",
    "run_custom_conditions",
    "run_single_condition_logits",
    "run_full_condition",
    "mask_checks",
    "tensor_summary",
    "score_record",
}


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=json_default) + "\n")


def sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files(model_path: str) -> tuple[Path, Path, list[Path]]:
    root = Path(model_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Model checkpoint path does not exist: {root}")
    if root.is_file():
        files = [root]
        base = root.parent
    else:
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and not {".cache", ".git"}.intersection(path.parts)
        )
        base = root
    if not files:
        raise RuntimeError(f"Model checkpoint contains no files: {root}")
    return root, base, files


def checkpoint_identity(model_path: str) -> dict[str, Any]:
    root, base, files = checkpoint_files(model_path)
    records = []
    for path in files:
        records.append(
            {
                "relative_path": str(path.relative_to(base)),
                "size": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return {
        "resolved_path": str(root),
        "file_count": len(records),
        "files": records,
        "content_sha256": sha256_json(records),
    }


def checkpoint_file_stats(model_path: str) -> list[dict[str, Any]]:
    _, base, files = checkpoint_files(model_path)
    return [
        {
            "relative_path": str(path.relative_to(base)),
            "size": int(path.stat().st_size),
            "mtime_ns": int(path.stat().st_mtime_ns),
        }
        for path in files
    ]


def verify_checkpoint_identity_quick(model_path: str, identity: dict[str, Any]) -> None:
    root, base, files = checkpoint_files(model_path)
    if str(root) != identity["resolved_path"]:
        raise RuntimeError(f"Model path changed after manifest creation: {root}")
    current_relative_paths = {str(path.relative_to(base)) for path in files}
    expected_relative_paths = {item["relative_path"] for item in identity["files"]}
    if current_relative_paths != expected_relative_paths:
        raise RuntimeError("Model checkpoint file set changed after manifest creation")
    for item in identity["files"]:
        path = base / item["relative_path"]
        if not path.is_file() or int(path.stat().st_size) != int(item["size"]):
            raise RuntimeError(f"Model checkpoint file changed after manifest creation: {path}")
        if int(item["size"]) <= 16 * 1024 * 1024 and sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Model checkpoint metadata file changed: {path}")


def source_tsv_path(lmu_data: str, dataset_name: str) -> Path:
    path = Path(lmu_data).expanduser().resolve() / f"{dataset_name}.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset source TSV does not exist: {path}")
    return path


def scoring_contract_sha256_from_source(source: str) -> str:
    tree = ast.parse(source)
    selected_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in SCORING_CONTRACT_FUNCTIONS:
                selected_nodes.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            if names & SCORING_CONTRACT_GLOBALS:
                selected_nodes.append(node)
    function_names = {
        node.name
        for node in selected_nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = SCORING_CONTRACT_FUNCTIONS - function_names
    if missing:
        raise RuntimeError(f"Scoring-contract functions missing from source: {sorted(missing)}")
    contract = ast.Module(body=selected_nodes, type_ignores=[])
    return hashlib.sha256(ast.dump(contract, include_attributes=False).encode()).hexdigest()


def current_scoring_contract_sha256() -> str:
    return scoring_contract_sha256_from_source(Path(__file__).read_text(encoding="utf-8"))


def is_present(value: Any) -> bool:
    if value is None:
        return False
    try:
        import pandas as pd

        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() not in {"", "nan", "none", "null"}


def embedded_choice_marker_sequence(question: str) -> list[str]:
    text = str(question or "")
    found = re.findall(r"(?:^|\n)\s*\(?([A-Z])\)?[\.)\:]\s*", text)
    if not found:
        found = re.findall(r"\(([A-Z])\)", text)
    return found


def embedded_choice_marker_labels(question: str) -> list[str]:
    return sorted(set(embedded_choice_marker_sequence(question)))


def embedded_choice_labels(question: str) -> list[str]:
    found = set(embedded_choice_marker_labels(question))
    labels = []
    for code in range(ord("A"), ord("Z") + 1):
        label = chr(code)
        if label not in found:
            break
        labels.append(label)
    return labels


def is_nonempty_option(value: Any) -> bool:
    if value is None:
        return False
    try:
        import pandas as pd

        if pd.isna(value):
            return False
    except Exception:
        pass
    return bool(str(value).strip())


def present_choice_labels(dataset_name: str, row: Any) -> list[str]:
    if dataset_name == "DynaMath":
        return embedded_choice_marker_sequence(str(row.get("question", "")))
    return [
        label
        for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if label in row.index and is_nonempty_option(row.get(label))
    ]


def all_single_choice_labels(dataset_name: str, row: Any) -> list[str]:
    present = set(present_choice_labels(dataset_name, row))
    labels = []
    for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if label not in present:
            break
        labels.append(label)
    return labels


def row_choice_labels(dataset_name: str, row: Any) -> list[str]:
    if dataset_name == "DynaMath":
        return embedded_choice_labels(str(row.get("question", "")))
    labels = []
    for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if label not in row.index or not is_present(row.get(label)):
            break
        labels.append(label)
    return labels


def row_option_texts(dataset_name: str, row: Any, labels: list[str]) -> dict[str, str]:
    if dataset_name != "DynaMath":
        return {label: str(row[label]) for label in labels}
    text = str(row.get("question", ""))
    line_pattern = r"(?:^|\n)\s*\(?([A-Z])\)?[\.)\:]\s*"
    matches = list(re.finditer(line_pattern, text))
    if not matches:
        matches = list(re.finditer(r"\(([A-Z])\)\s*", text))
    options: dict[str, str] = {}
    for idx, match in enumerate(matches):
        label = match.group(1)
        if label not in labels:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        options[label] = text[start:end].strip()
    return options


def normalized_option_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def normalize_answer(value: Any) -> str:
    match = re.search(r"[A-Z]", str(value or "").upper())
    return match.group(0) if match else ""


def strict_answer_label(value: Any) -> str:
    match = re.fullmatch(r"\s*([A-Z])\s*", str(value or ""))
    return match.group(1) if match else ""


def selected_subset_records(
    dataset_name: str,
    data: Any,
    selection_profile: str = "fixed_choice",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if dataset_name not in DATASET_TARGETS:
        raise KeyError(f"Unsupported single-choice dataset: {dataset_name}")
    if selection_profile not in SELECTION_PROFILES:
        raise ValueError(f"Unknown selection profile: {selection_profile}")
    if selection_profile == "all_single_choice" and dataset_name not in ALL_SINGLE_CHOICE_DATASETS:
        raise ValueError(f"Dataset does not support all-single-choice selection: {dataset_name}")
    target = (
        int(DATASET_TARGETS[dataset_name]["choice_count"])
        if selection_profile == "fixed_choice"
        else None
    )
    mmbench_group_sizes: Counter[int] = Counter()
    if dataset_name == "MMBench_DEV_EN_V11":
        mmbench_group_sizes.update(int(row["index"]) % 1_000_000 for _, row in data.iterrows())

    selected: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    for row_position, row in data.iterrows():
        labels = (
            row_choice_labels(dataset_name, row)
            if selection_profile == "fixed_choice"
            else all_single_choice_labels(dataset_name, row)
        )
        answer = (
            normalize_answer(row.get("answer"))
            if selection_profile == "fixed_choice"
            else strict_answer_label(row.get("answer"))
        )
        if dataset_name == "DynaMath" and str(row.get("answer_type", "")) != "multiple choice":
            rejection_counts["not_multiple_choice"] += 1
            continue
        if selection_profile == "fixed_choice":
            if labels != list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: int(target)]):
                rejection_counts[f"not_{target}_consecutive_choices"] += 1
                continue
        else:
            present_labels = present_choice_labels(dataset_name, row)
            if len(labels) < 2:
                rejection_counts["fewer_than_2_choices"] += 1
                continue
            if present_labels != labels:
                rejection_counts["nonconsecutive_choice_columns"] += 1
                continue
            if not answer:
                rejection_counts["answer_not_strict_single_label"] += 1
                continue
        if answer not in labels:
            rejection_counts["answer_outside_labels"] += 1
            continue
        group_size = None
        if dataset_name == "MMBench_DEV_EN_V11":
            raw_index = int(row["index"])
            base_index = raw_index % 1_000_000
            group_size = int(mmbench_group_sizes[base_index])
            if raw_index >= 1_000_000:
                rejection_counts["circular_noncanonical"] += 1
                continue
        options = row_option_texts(dataset_name, row, labels)
        if set(options) != set(labels):
            rejection_counts["option_parse_failure"] += 1
            continue
        normalized_options = [normalized_option_text(options[label]) for label in labels]
        if dataset_name != "DynaMath":
            if any(not value for value in normalized_options):
                rejection_counts["empty_option_text"] += 1
                continue
            if len(set(normalized_options)) != len(normalized_options):
                rejection_counts["duplicate_option_text"] += 1
                continue
        elif selection_profile == "all_single_choice":
            nonempty_options = [value for value in normalized_options if value]
            if len(set(nonempty_options)) != len(nonempty_options):
                rejection_counts["duplicate_option_text"] += 1
                continue
        selected.append(
            {
                "dataset": dataset_name,
                "row_position": int(row_position),
                "sample_index": str(row.get("index")),
                "answer_key": answer,
                "choice_labels": labels,
                "choice_count": len(labels),
                "option_text_sha256": sha256_json(options),
                "circular_group_size": group_size,
            }
        )

    selected.sort(key=lambda item: item["row_position"])
    summary = {
        "dataset": dataset_name,
        "subset": (
            DATASET_TARGETS[dataset_name]["subset"]
            if selection_profile == "fixed_choice"
            else DATASET_TARGETS[dataset_name]["all_subset"]
        ),
        "selection_profile": selection_profile,
        "target_choice_count": target,
        "total_rows": int(len(data)),
        "selected_rows": len(selected),
        "selected_choice_count_histogram": dict(
            Counter(item["choice_count"] for item in selected)
        ),
        "answer_histogram": dict(Counter(item["answer_key"] for item in selected)),
        "rejection_counts": dict(rejection_counts),
        "selected_circular_group_size_histogram": dict(
            Counter(item["circular_group_size"] for item in selected if item["circular_group_size"] is not None)
        ),
        "unique_sample_indices": len({item["sample_index"] for item in selected}),
    }
    return selected, summary


def build_runtime(
    repo_root: Path,
    output_root: Path,
    dataset_name: str,
    model_key: str,
    gpu_id: str,
    model_path: str,
    lmu_data: str,
    matrix_config: str,
):
    return runtime_env(
        repo_root,
        output_root,
        model_key,
        dataset_name,
        REPLAY_MODE,
        POLICY,
        gpu_id,
        model_path_override=model_path,
        lmu_data=lmu_data,
        matrix_config=matrix_config,
    )


def make_manifest(args: argparse.Namespace) -> dict[str, Any]:
    from vlmeval.dataset import build_dataset

    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    matrix_config = Path(args.matrix_config or repo_root / DEFAULT_MATRIX_CONFIG).resolve()
    models_config = repo_root / "configs" / "models.yaml"
    runtime_root = output.parent / "_manifest_runtime"
    records: list[dict[str, Any]] = []
    dataset_summaries = []
    source_data: dict[str, dict[str, Any]] = {}
    registry_names: set[str] = set()
    resolved_model_paths: set[str] = set()
    for dataset_name in args.datasets:
        env, runner, _ = build_runtime(
            repo_root,
            runtime_root,
            dataset_name,
            args.model_key,
            args.gpu_id,
            args.model_path,
            args.lmu_data,
            str(matrix_config),
        )
        registry_names.add(runner.models[args.model_key].registry_name)
        resolved_model_paths.add(str(Path(env["MODEL_PATH"]).expanduser().resolve()))
        with patched_environ(env):
            dataset = build_dataset(dataset_name)
            selected, summary = selected_subset_records(
                dataset_name,
                dataset.data,
                selection_profile=args.selection_profile,
            )
        source_path = source_tsv_path(args.lmu_data, dataset_name)
        source_entry = {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "size": int(source_path.stat().st_size),
        }
        source_data[dataset_name] = source_entry
        summary["source_tsv"] = source_entry
        for position, item in enumerate(selected):
            item["dataset_position"] = position
            item["shard"] = position % int(args.num_shards)
        records.extend(selected)
        dataset_summaries.append(summary)

    if len(registry_names) != 1:
        raise RuntimeError(
            f"Manifest datasets resolved multiple model registries: {registry_names}"
        )
    if len(resolved_model_paths) != 1:
        raise RuntimeError(
            f"Manifest datasets resolved multiple checkpoints: {resolved_model_paths}"
        )
    registry_name = next(iter(registry_names))
    model_identity = checkpoint_identity(next(iter(resolved_model_paths)))

    payload = {
        "schema": (
            "topic-image-replay/readout-v2-fixed-choice/v1"
            if args.selection_profile == "fixed_choice"
            else "topic-image-replay/readout-v2-all-single-choice/v1"
        ),
        "created_at": time.time(),
        "repo_root": str(repo_root),
        "git_base": "67764a929ebd0837e4cbf776739394faf6001ed2",
        "repo_snapshot": repo_git_snapshot(repo_root),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scoring_contract_sha256": current_scoring_contract_sha256(),
        "selection_profile": args.selection_profile,
        "matrix_config": str(matrix_config),
        "matrix_config_sha256": hashlib.sha256(matrix_config.read_bytes()).hexdigest(),
        "models_config": str(models_config),
        "models_config_sha256": sha256_file(models_config),
        "model_key": args.model_key,
        "registry_name": registry_name,
        "model_identity": model_identity,
        "model_identity_sha256": model_identity["content_sha256"],
        "source_data": source_data,
        "source_data_sha256": sha256_json(source_data),
        "replay_mode": REPLAY_MODE,
        "policy": POLICY,
        "answer_prefix": ANSWER_PREFIX,
        "conditions": list(CONDITIONS),
        "num_shards": int(args.num_shards),
        "datasets": dataset_summaries,
        "record_count": len(records),
        "records_sha256": sha256_json(records),
        "records": records,
    }
    write_json(output, payload)
    return payload


def token_id(tokenizer: Any, token: str) -> int | None:
    value = tokenizer.convert_tokens_to_ids(token)
    unk = getattr(tokenizer, "unk_token_id", None)
    if not isinstance(value, int) or value < 0 or value == unk:
        return None
    return int(value)


def contiguous_token_spans(ids: list[int], target: int) -> list[dict[str, int]]:
    spans = []
    start = None
    for idx, value in enumerate(ids):
        if value == target and start is None:
            start = idx
        elif value != target and start is not None:
            spans.append({"start": start, "end": idx - 1})
            start = None
    if start is not None:
        spans.append({"start": start, "end": len(ids) - 1})
    return spans


def qwen_image_spans(input_ids: torch.Tensor, tokenizer: Any) -> list[dict[str, int]]:
    ids = input_ids[0].detach().cpu().tolist()
    image_id = token_id(tokenizer, "<|image_pad|>")
    if image_id is None:
        raise RuntimeError("Qwen tokenizer has no <|image_pad|> token")
    core_spans = contiguous_token_spans(ids, image_id)
    start_id = token_id(tokenizer, "<|vision_start|>")
    end_id = token_id(tokenizer, "<|vision_end|>")
    spans = []
    for position, core in enumerate(core_spans, start=1):
        start = core["start"]
        end = core["end"]
        expanded_start = start - 1 if start > 0 and ids[start - 1] == start_id else start
        expanded_end = end + 1 if end + 1 < len(ids) and ids[end + 1] == end_id else end
        spans.append(
            {
                "image_position": position,
                "start": expanded_start,
                "end": expanded_end,
                "core_start": start,
                "core_end": end,
            }
        )
    return spans


def longest_common_prefix(sequences: list[list[int]]) -> list[int]:
    if not sequences:
        return []
    limit = min(len(item) for item in sequences)
    count = 0
    while count < limit and len({item[count] for item in sequences}) == 1:
        count += 1
    return sequences[0][:count]


def candidate_token_plan(tokenizer: Any, labels: list[str], answer_prefix: str = ANSWER_PREFIX) -> dict[str, Any]:
    joint = {
        label: list(tokenizer(answer_prefix + label, add_special_tokens=False).input_ids)
        for label in labels
    }
    forced_prefix_ids = longest_common_prefix(list(joint.values()))
    if not forced_prefix_ids:
        raise RuntimeError(f"No common forced-prefix tokens for labels {labels}")
    suffixes = {label: ids[len(forced_prefix_ids) :] for label, ids in joint.items()}
    if any(len(ids) != 1 for ids in suffixes.values()):
        raise RuntimeError(f"Readout v2 requires one-token labels after shared prefix: {suffixes}")
    candidate_ids = {label: int(ids[0]) for label, ids in suffixes.items()}
    if len(set(candidate_ids.values())) != len(labels):
        raise RuntimeError(f"Candidate token ids are not distinct: {candidate_ids}")
    return {
        "answer_prefix_requested": answer_prefix,
        "forced_prefix_ids": forced_prefix_ids,
        "forced_prefix_text": tokenizer.decode(
            forced_prefix_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "candidate_token_ids": candidate_ids,
        "candidate_token_text": {
            label: tokenizer.decode(
                [value],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for label, value in candidate_ids.items()
        },
        "joint_token_ids": joint,
    }


def extend_sequence_inputs(inputs: dict[str, Any], extra_ids: list[int]) -> dict[str, Any]:
    out = dict(inputs)
    input_ids = inputs["input_ids"]
    seq_len = int(input_ids.shape[-1])
    extra = torch.tensor([extra_ids], dtype=input_ids.dtype, device=input_ids.device)
    out["input_ids"] = torch.cat([input_ids, extra], dim=-1)
    for key, value in inputs.items():
        if key == "input_ids" or not isinstance(value, torch.Tensor):
            continue
        if value.ndim == 2 and int(value.shape[-1]) == seq_len:
            if key != "attention_mask":
                raise RuntimeError(f"Unknown sequence tensor requires explicit extension semantics: {key}")
            fill = torch.ones((value.shape[0], len(extra_ids)), dtype=value.dtype, device=value.device)
            out[key] = torch.cat([value, fill], dim=-1)
    return out


def derive_prefill_boundary(
    wrapper: Any,
    messages: list[dict[str, Any]],
    prompt_text: Any,
    generation_text: Any,
    generation_inputs: dict[str, Any],
    images: Any,
    videos: Any,
) -> tuple[int, dict[str, Any]]:
    tokenizer = wrapper.processor.tokenizer
    ids = generation_inputs["input_ids"][0].detach().cpu().tolist()
    prompt_value = prompt_text[0] if isinstance(prompt_text, list) and len(prompt_text) == 1 else prompt_text
    generation_value = (
        generation_text[0]
        if isinstance(generation_text, list) and len(generation_text) == 1
        else generation_text
    )
    if not isinstance(prompt_value, str) or not isinstance(generation_value, str):
        raise RuntimeError(
            f"Expected one rendered conversation, got {type(prompt_text).__name__} and "
            f"{type(generation_text).__name__}"
        )
    suffix_ok = generation_value.startswith(prompt_value)
    suffix_text = generation_value[len(prompt_value) :] if suffix_ok else ""
    suffix_ids = list(tokenizer(suffix_text, add_special_tokens=False).input_ids) if suffix_text else []
    token_suffix_ok = bool(suffix_ids) and ids[-len(suffix_ids) :] == suffix_ids
    method = "chat_template_suffix"
    if token_suffix_ok:
        boundary = len(ids) - len(suffix_ids)
    else:
        prompt_inputs = wrapper.processor(
            text=prompt_text,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        prompt_ids = prompt_inputs["input_ids"][0].tolist()
        if ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError("Generation-prompt input is not prefixed by the no-generation prompt")
        boundary = len(prompt_ids)
        method = "processor_crosscheck_fallback"
    return boundary, {
        "method": method,
        "prompt_text_prefix_ok": suffix_ok,
        "assistant_seed_text": suffix_text,
        "assistant_seed_token_ids": suffix_ids,
        "assistant_seed_token_suffix_ok": token_suffix_ok,
    }


def prepare_inputs(wrapper: Any, dataset: Any, dataset_name: str, row: Any) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    prompt = build_standard_prompt(wrapper, dataset, dataset_name, row)
    hf_content = wrapper._prepare_content(prompt, dataset=dataset_name)
    messages = []
    if getattr(wrapper, "system_prompt", None) is not None:
        messages.append({"role": "system", "content": wrapper.system_prompt})
    messages.append({"role": "user", "content": hf_content})
    prompt_text = wrapper.processor.apply_chat_template(
        [messages], tokenize=False, add_generation_prompt=False
    )
    generation_text = wrapper.processor.apply_chat_template(
        [messages], tokenize=False, add_generation_prompt=True
    )
    images, videos = process_vision_info([messages])
    generation_inputs = wrapper.processor(
        text=generation_text,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    generation_inputs = dict(generation_inputs)
    prefill_len, boundary_meta = derive_prefill_boundary(
        wrapper,
        messages,
        prompt_text,
        generation_text,
        generation_inputs,
        images,
        videos,
    )
    if not torch.all(generation_inputs["attention_mask"] == 1):
        raise RuntimeError("Single-sample readout path unexpectedly contains padding")
    spans = qwen_image_spans(generation_inputs["input_ids"], wrapper.processor.tokenizer)
    if len(spans) != 2:
        raise RuntimeError(f"IQI must contain exactly two image-token spans, got {spans}")
    if spans[1]["end"] >= prefill_len:
        raise RuntimeError(f"I2 span crosses the prefill boundary: {spans[1]} vs {prefill_len}")
    counts = Counter(str(item.get("type")) for item in hf_content)
    if counts.get("image", 0) != 2 or counts.get("text", 0) != 1:
        raise RuntimeError(f"Main-entry replay did not produce IQI content: {counts}")
    return {
        "inputs": generation_inputs,
        "prefill_len": prefill_len,
        "image_spans": spans,
        "standard_prompt": prompt,
        "hf_content": hf_content,
        "messages": messages,
        "prompt_text": prompt_text,
        "generation_text": generation_text,
        "boundary_meta": boundary_meta,
        "content_counts": dict(counts),
    }


def allowed_masks(seq_len: int, decode_start: int, i2_span: dict[str, int]) -> torch.Tensor:
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    baseline = causal.clone()
    baseline[decode_start:, :decode_start] = False
    readout = baseline.clone()
    readout[decode_start:, int(i2_span["start"]) : int(i2_span["end"]) + 1] = True
    return torch.stack([baseline, readout, causal], dim=0)


def additive_mask(allowed: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    allowed = allowed.to(device=device)
    mask = torch.full(allowed.shape, torch.finfo(dtype).min, dtype=dtype, device=device)
    mask.masked_fill_(allowed, 0)
    return mask.unsqueeze(1)


def repeat_qwen_inputs(inputs: dict[str, Any], batch_size: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            out[key] = value
            continue
        if key in {"input_ids", "attention_mask", "token_type_ids"}:
            repeats = [batch_size] + [1] * (value.ndim - 1)
            out[key] = value.repeat(*repeats)
        elif key in {
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "second_per_grid_ts",
        }:
            repeats = [batch_size] + [1] * (value.ndim - 1)
            out[key] = value.repeat(*repeats)
        else:
            raise RuntimeError(f"Unknown Qwen tensor cannot be safely repeated: {key} {tuple(value.shape)}")
    return out


def install_qwen_position_ids(model: Any, inputs: dict[str, Any], attention_mask_2d: torch.Tensor) -> dict[str, Any]:
    rope_model = getattr(model, "model", None)
    get_rope_index = getattr(rope_model, "get_rope_index", None)
    if get_rope_index is None:
        raise RuntimeError("Qwen model does not expose get_rope_index")
    position_ids, rope_deltas = get_rope_index(
        inputs.get("input_ids"),
        inputs.get("image_grid_thw"),
        inputs.get("video_grid_thw"),
        second_per_grid_ts=inputs.get("second_per_grid_ts"),
        attention_mask=attention_mask_2d,
    )
    inputs["position_ids"] = position_ids
    inputs["rope_deltas"] = rope_deltas
    try:
        rope_model.rope_deltas = rope_deltas
    except Exception:
        pass
    return {"position_ids": position_ids, "rope_deltas": rope_deltas}


def candidate_scores(logits: torch.Tensor, plan: dict[str, Any]) -> dict[str, float]:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return {
        label: float(log_probs[int(token)].detach().cpu().item())
        for label, token in plan["candidate_token_ids"].items()
    }


def score_from_values(values: dict[str, float], answer_key: str) -> dict[str, Any]:
    predicted = max(values, key=values.get)
    return {
        "candidate_logprobs": values,
        "predicted_key": predicted,
        "answer_key": answer_key,
        "hit": predicted == answer_key,
    }


def clone_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}


def corrupt_blocked_question_tokens(
    inputs: dict[str, Any],
    tokenizer: Any,
    image_spans: list[dict[str, int]],
    decode_start: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = clone_inputs(inputs)
    ids = out["input_ids"][0]
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    start = int(image_spans[0]["end"]) + 1
    end = min(int(image_spans[1]["start"]), int(decode_start))
    replacement_ids = tokenizer(" unrelated", add_special_tokens=False).input_ids
    replacement = int(replacement_ids[-1])
    positions = [idx for idx in range(start, end) if int(ids[idx]) not in special][:16]
    if not positions:
        raise RuntimeError("Could not find safe Q1 tokens for blocked-context corruption")
    before = [int(ids[idx]) for idx in positions]
    for idx in positions:
        ids[idx] = replacement
    return out, {
        "positions": positions,
        "before_ids": before,
        "replacement_id": replacement,
        "replacement_text": tokenizer.decode([replacement], skip_special_tokens=False),
    }


def model_dtype(model: Any) -> torch.dtype:
    value = getattr(model, "dtype", None)
    if isinstance(value, torch.dtype) and value.is_floating_point:
        return value
    return next(model.parameters()).dtype


def run_custom_conditions(
    model: Any,
    inputs: dict[str, Any],
    allowed: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    custom_inputs = repeat_qwen_inputs(inputs, int(allowed.shape[0]))
    attention_mask_2d = custom_inputs["attention_mask"].clone()
    position = install_qwen_position_ids(model, custom_inputs, attention_mask_2d)
    custom_inputs["attention_mask"] = additive_mask(
        allowed,
        dtype=model_dtype(model),
        device=custom_inputs["input_ids"].device,
    )
    with torch.inference_mode():
        outputs = model(**custom_inputs, use_cache=False, return_dict=True)
    return outputs.logits[:, -1, :], position


def run_single_condition_logits(
    model: Any,
    inputs: dict[str, Any],
    allowed: torch.Tensor,
) -> torch.Tensor:
    logits, _ = run_custom_conditions(model, inputs, allowed.unsqueeze(0))
    return logits[0]


def run_full_condition(model: Any, inputs: dict[str, Any]) -> torch.Tensor:
    full_inputs = clone_inputs(inputs)
    try:
        model.rope_deltas = None
        if getattr(model, "model", None) is not None:
            model.model.rope_deltas = None
    except Exception:
        pass
    with torch.inference_mode():
        outputs = model(**full_inputs, use_cache=False, return_dict=True)
    return outputs.logits[0, -1, :]


def mask_checks(allowed: torch.Tensor, decode_start: int, i2_span: dict[str, int]) -> dict[str, Any]:
    seq_len = int(allowed.shape[-1])
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    expected_baseline = causal.clone()
    expected_baseline[decode_start:, :decode_start] = False
    expected_readout = expected_baseline.clone()
    expected_readout[decode_start:, int(i2_span["start"]) : int(i2_span["end"]) + 1] = True
    return {
        "prefill_causal_baseline": bool(torch.equal(allowed[0, :decode_start], causal[:decode_start])),
        "prefill_causal_readout_v2": bool(torch.equal(allowed[1, :decode_start], causal[:decode_start])),
        "baseline_exact": bool(torch.equal(allowed[0], expected_baseline)),
        "readout_v2_exact": bool(torch.equal(allowed[1], expected_readout)),
        "full_exact": bool(torch.equal(allowed[2], causal)),
        "baseline_decode_prompt_allowed": int(allowed[0, decode_start:, :decode_start].sum().item()),
        "readout_decode_prompt_allowed": int(allowed[1, decode_start:, :decode_start].sum().item()),
        "readout_expected_i2_per_row": int(i2_span["end"] - i2_span["start"] + 1),
        "no_future_baseline": bool(torch.equal(allowed[0] & ~causal, torch.zeros_like(causal))),
        "no_future_readout_v2": bool(torch.equal(allowed[1] & ~causal, torch.zeros_like(causal))),
        "no_future_full": bool(torch.equal(allowed[2] & ~causal, torch.zeros_like(causal))),
    }


def tensor_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__}
    cpu = value.detach().cpu().contiguous()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
    }


def score_record(
    wrapper: Any,
    dataset: Any,
    dataset_name: str,
    row: Any,
    manifest_record: dict[str, Any],
    dump_dir: Path | None,
    diagnostics: bool,
) -> dict[str, Any]:
    prepared = prepare_inputs(wrapper, dataset, dataset_name, row)
    tokenizer = wrapper.processor.tokenizer
    plan = candidate_token_plan(tokenizer, list(manifest_record["choice_labels"]))
    extended = extend_sequence_inputs(prepared["inputs"], plan["forced_prefix_ids"])
    seq_len = int(extended["input_ids"].shape[-1])
    decode_start = int(prepared["prefill_len"])
    image2_envelope = prepared["image_spans"][1]
    i2_span = {
        "start": int(image2_envelope["core_start"]),
        "end": int(image2_envelope["core_end"]),
        "span_kind": "image2_visual_core",
    }
    allowed = allowed_masks(seq_len, decode_start, i2_span)
    checks = mask_checks(allowed, decode_start, i2_span)
    if not all(
        checks[key]
        for key in (
            "prefill_causal_baseline",
            "prefill_causal_readout_v2",
            "baseline_exact",
            "readout_v2_exact",
            "full_exact",
            "no_future_baseline",
            "no_future_readout_v2",
            "no_future_full",
        )
    ):
        raise RuntimeError(f"Mask construction invariant failed: {checks}")

    torch.cuda.synchronize()
    start = time.perf_counter()
    custom_logits, position = run_custom_conditions(wrapper.model, extended, allowed)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    raw_logits = {
        "baseline": custom_logits[0],
        "readout_v2": custom_logits[1],
        "full": custom_logits[2],
    }
    conditions = {
        condition: score_from_values(candidate_scores(raw_logits[condition], plan), manifest_record["answer_key"])
        for condition in CONDITIONS
    }
    diagnostic_payload: dict[str, Any] = {}
    single_condition_logits: torch.Tensor | None = None
    if diagnostics:
        full2d = run_full_condition(wrapper.model, extended)
        candidate_ids = list(plan["candidate_token_ids"].values())
        parity_diff = (custom_logits[2, candidate_ids].float() - full2d[candidate_ids].float()).abs()
        corrupted, corruption_meta = corrupt_blocked_question_tokens(
            extended,
            tokenizer,
            prepared["image_spans"],
            decode_start,
        )
        corrupted_logits, _ = run_custom_conditions(wrapper.model, corrupted, allowed)
        baseline_diff = (
            custom_logits[0, candidate_ids].float() - corrupted_logits[0, candidate_ids].float()
        ).abs()
        readout_diff = (
            custom_logits[1, candidate_ids].float() - corrupted_logits[1, candidate_ids].float()
        ).abs()
        single_condition_logits = torch.stack(
            [
                run_single_condition_logits(wrapper.model, extended, allowed[index])
                for index in range(len(CONDITIONS))
            ],
            dim=0,
        )
        parity_rows = {}
        for index, condition in enumerate(CONDITIONS):
            candidate_diff = (
                custom_logits[index, candidate_ids].float()
                - single_condition_logits[index, candidate_ids].float()
            ).abs()
            batched_argmax = max(
                plan["candidate_token_ids"],
                key=lambda label: float(
                    custom_logits[index, plan["candidate_token_ids"][label]].item()
                ),
            )
            single_argmax = max(
                plan["candidate_token_ids"],
                key=lambda label: float(
                    single_condition_logits[index, plan["candidate_token_ids"][label]].item()
                ),
            )
            parity_rows[condition] = {
                "candidate_logit_max_abs_diff": float(candidate_diff.max().item()),
                "candidate_argmax_equal": batched_argmax == single_argmax,
                "batched_argmax": batched_argmax,
                "single_argmax": single_argmax,
            }
        diagnostic_payload = {
            "full_4d_vs_standard_2d_candidate_logit_max_abs_diff": float(parity_diff.max().item()),
            "full_4d_candidate_argmax": max(
                plan["candidate_token_ids"],
                key=lambda label: float(custom_logits[2, plan["candidate_token_ids"][label]].item()),
            ),
            "standard_2d_candidate_argmax": max(
                plan["candidate_token_ids"],
                key=lambda label: float(full2d[plan["candidate_token_ids"][label]].item()),
            ),
            "baseline_blocked_corruption_candidate_logit_max_abs_diff": float(baseline_diff.max().item()),
            "baseline_blocked_corruption_pass_1e_5": bool(baseline_diff.max().item() <= 1e-5),
            "readout_v2_corruption_candidate_logit_max_abs_diff": float(readout_diff.max().item()),
            "batch3_vs_single4d": parity_rows,
            "corruption": corruption_meta,
        }
        diagnostic_payload["full_4d_vs_standard_2d_candidate_argmax_equal"] = (
            diagnostic_payload["full_4d_candidate_argmax"]
            == diagnostic_payload["standard_2d_candidate_argmax"]
        )
        if not diagnostic_payload["baseline_blocked_corruption_pass_1e_5"]:
            raise RuntimeError(f"Blocked-context corruption leaked into baseline: {diagnostic_payload}")

    record = {
        "schema": "topic-image-replay/readout-v2-record/v1",
        "dataset": dataset_name,
        "row_position": int(manifest_record["row_position"]),
        "sample_index": str(manifest_record["sample_index"]),
        "answer_key": manifest_record["answer_key"],
        "choice_labels": manifest_record["choice_labels"],
        "conditions": conditions,
        "timing_seconds": elapsed,
        "token_count": seq_len,
        "prefill_token_count": decode_start,
        "decode_seed_and_forced_prefix_token_count": seq_len - decode_start,
        "image_spans": prepared["image_spans"],
        "readout_v2_span": i2_span,
        "candidate_plan": plan,
        "mask_checks": checks,
        "diagnostics": diagnostic_payload,
    }

    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        np.save(dump_dir / "input_ids.npy", extended["input_ids"].detach().cpu().numpy())
        np.save(dump_dir / "allowed_masks_baseline_readout_v2_full.npy", allowed.cpu().numpy())
        np.save(dump_dir / "position_ids_custom.npy", position["position_ids"].detach().cpu().numpy())
        stacked_logits = torch.stack([raw_logits[name].float().cpu() for name in CONDITIONS], dim=0)
        np.save(dump_dir / "last_token_logits_baseline_readout_v2_full.npy", stacked_logits.numpy())
        if single_condition_logits is not None:
            np.save(
                dump_dir / "last_token_logits_single4d_baseline_readout_v2_full.npy",
                single_condition_logits.float().cpu().numpy(),
            )
        ids = extended["input_ids"][0].detach().cpu().tolist()
        token_table = [
            {
                "position": idx,
                "token_id": int(value),
                "token": tokenizer.decode(
                    [int(value)], skip_special_tokens=False, clean_up_tokenization_spaces=False
                ),
            }
            for idx, value in enumerate(ids)
        ]
        artifact = {
            **record,
            "standard_prompt": summarize_prompt_items(prepared["standard_prompt"]),
            "post_template_replay_content": summarize_prompt_items(prepared["hf_content"]),
            "content_counts": prepared["content_counts"],
            "prompt_text": prepared["prompt_text"],
            "generation_text": prepared["generation_text"],
            "boundary_meta": prepared["boundary_meta"],
            "token_table": token_table,
            "input_tensors": {key: tensor_summary(value) for key, value in extended.items()},
            "raw_files": {
                "input_ids": "input_ids.npy",
                "allowed_masks": "allowed_masks_baseline_readout_v2_full.npy",
                "position_ids": "position_ids_custom.npy",
                "last_token_logits": "last_token_logits_baseline_readout_v2_full.npy",
                "last_token_logits_single4d": (
                    "last_token_logits_single4d_baseline_readout_v2_full.npy"
                    if single_condition_logits is not None
                    else None
                ),
            },
        }
        write_json(dump_dir / "artifact.json", artifact)
    return record


def bind_dump_artifact(
    dump_dir: Path,
    record: dict[str, Any],
    manifest_record: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    artifact_path = dump_dir / "artifact.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    for field in (
        "dataset",
        "row_position",
        "sample_index",
        "answer_key",
        "choice_labels",
        "conditions",
        "mask_checks",
    ):
        if artifact.get(field) != record.get(field):
            raise RuntimeError(f"Raw artifact/JSONL mismatch before binding: field={field}")
    artifact["runtime"] = record["runtime"]
    artifact["provenance"] = record["provenance"]
    artifact["manifest_binding"] = {
        "records_sha256": manifest["records_sha256"],
        "manifest_record_sha256": sha256_json(manifest_record),
        "model_identity_sha256": manifest.get("model_identity_sha256"),
        "source_data_sha256": manifest.get("source_data_sha256"),
    }
    artifact["jsonl_record_sha256"] = sha256_json(record)
    write_json(artifact_path, artifact)


def expected_provenance(manifest: dict[str, Any]) -> dict[str, str]:
    provenance = {
        "records_sha256": str(manifest["records_sha256"]),
        "implementation_sha256": str(manifest["implementation_sha256"]),
        "matrix_config_sha256": str(manifest["matrix_config_sha256"]),
        "repo_head": str(manifest["repo_snapshot"]["head"]),
    }
    if manifest.get("scoring_contract_sha256"):
        provenance["scoring_contract_sha256"] = str(manifest["scoring_contract_sha256"])
    for field in (
        "models_config_sha256",
        "model_identity_sha256",
        "source_data_sha256",
        "registry_name",
    ):
        if manifest.get(field):
            provenance[field] = str(manifest[field])
    return provenance


def validate_prediction_payload(
    row: dict[str, Any],
    expected: dict[str, Any],
    provenance: dict[str, str],
    *,
    expected_shard_rank: int | None = None,
) -> None:
    key = record_key(row)
    if row.get("schema") != "topic-image-replay/readout-v2-record/v1":
        raise RuntimeError(f"Prediction schema mismatch for {key}: {row.get('schema')}")
    if row.get("provenance") != provenance:
        raise RuntimeError(
            f"Prediction provenance mismatch for {key}: "
            f"{row.get('provenance')} != {provenance}"
        )
    for field in ("row_position", "answer_key", "choice_labels"):
        if row.get(field) != expected.get(field):
            raise RuntimeError(
                f"Prediction/manifest field mismatch for {key} field={field}: "
                f"{row.get(field)} != {expected.get(field)}"
            )
    if set(row.get("conditions", {})) != set(CONDITIONS):
        raise RuntimeError(f"Prediction condition set mismatch for {key}")
    labels = list(expected["choice_labels"])
    answer = str(expected["answer_key"])
    for condition in CONDITIONS:
        values = row["conditions"][condition]
        logprobs = values.get("candidate_logprobs", {})
        if set(logprobs) != set(labels):
            raise RuntimeError(f"Candidate set mismatch for {key} condition={condition}")
        numeric = {label: float(logprobs[label]) for label in labels}
        if not all(math.isfinite(value) for value in numeric.values()):
            raise RuntimeError(f"Non-finite candidate score for {key} condition={condition}")
        predicted = max(labels, key=lambda label: numeric[label])
        if values.get("predicted_key") != predicted:
            raise RuntimeError(f"Predicted key is inconsistent for {key} condition={condition}")
        if values.get("answer_key") != answer:
            raise RuntimeError(f"Condition answer key mismatch for {key} condition={condition}")
        if not isinstance(values.get("hit"), bool) or values["hit"] != (predicted == answer):
            raise RuntimeError(f"Hit is inconsistent for {key} condition={condition}")
    required_mask_checks = (
        "prefill_causal_baseline",
        "prefill_causal_readout_v2",
        "baseline_exact",
        "readout_v2_exact",
        "full_exact",
        "no_future_baseline",
        "no_future_readout_v2",
        "no_future_full",
    )
    if not all(row.get("mask_checks", {}).get(name) is True for name in required_mask_checks):
        raise RuntimeError(f"Prediction mask checks failed for {key}")
    if expected_shard_rank is not None:
        if int(expected["shard"]) != int(expected_shard_rank):
            raise RuntimeError(
                f"Prediction is in the wrong shard for {key}: "
                f"{expected_shard_rank} != {expected['shard']}"
            )
        runtime_rank = row.get("runtime", {}).get("shard_rank")
        if int(runtime_rank) != int(expected_shard_rank):
            raise RuntimeError(f"Prediction runtime shard mismatch for {key}: {runtime_rank}")


def load_completed(
    path: Path,
    manifest: dict[str, Any],
    provenance: dict[str, str],
    *,
    expected_shard_rank: int | None,
) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    manifest_records = manifest_record_map(manifest)
    completed = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = record_key(row)
            if key in completed:
                raise RuntimeError(f"Resume shard contains duplicate key: {key}")
            if key not in manifest_records:
                raise RuntimeError(f"Resume shard key is absent from manifest: {key}")
            validate_prediction_payload(
                row,
                manifest_records[key],
                provenance,
                expected_shard_rank=expected_shard_rank,
            )
            completed.add(key)
    return completed


def validate_manifest_runtime_contract(
    manifest: dict[str, Any],
    args: argparse.Namespace,
    matrix_config: Path,
    *,
    require_attestation: bool = True,
) -> None:
    repo_root = Path(args.repo_root).resolve()
    if repo_root != Path(manifest["repo_root"]).resolve():
        raise RuntimeError("Runtime repository differs from the manifest repository")
    snapshot = repo_git_snapshot(repo_root)
    if snapshot["head"] != manifest["repo_snapshot"]["head"]:
        raise RuntimeError("Repository HEAD changed after manifest creation")
    if snapshot.get("status_short"):
        raise RuntimeError(f"Runtime repository is dirty: {snapshot['status_short']}")
    current_implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if current_implementation != manifest["implementation_sha256"]:
        raise RuntimeError("Readout implementation changed after manifest creation")
    if manifest.get("scoring_contract_sha256") != current_scoring_contract_sha256():
        raise RuntimeError("Scoring contract changed after manifest creation")
    if args.model_key != manifest["model_key"]:
        raise RuntimeError(f"Model key changed after manifest creation: {args.model_key}")
    if sha256_file(repo_root / "configs" / "models.yaml") != manifest["models_config_sha256"]:
        raise RuntimeError("Model registry changed after manifest creation")
    if sha256_file(matrix_config) != manifest["matrix_config_sha256"]:
        raise RuntimeError("Matrix config changed after manifest creation")
    runtime_model_path = args.model_path or manifest["model_identity"]["resolved_path"]
    verify_checkpoint_identity_quick(runtime_model_path, manifest["model_identity"])
    source_data = manifest["source_data"]
    if sha256_json(source_data) != manifest["source_data_sha256"]:
        raise RuntimeError("Manifest source-data identity is internally inconsistent")
    for dataset_name in args.datasets:
        path = source_tsv_path(args.lmu_data, dataset_name)
        expected = source_data[dataset_name]
        if str(path) != expected["path"] or sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"Dataset source changed after manifest creation: {dataset_name}")
    if not require_attestation:
        return
    attestation_path = Path(args.runtime_attestation).resolve()
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    expected_attestation = {
        "repo_head": manifest["repo_snapshot"]["head"],
        "implementation_sha256": manifest["implementation_sha256"],
        "model_identity_sha256": manifest["model_identity_sha256"],
        "source_data_sha256": manifest["source_data_sha256"],
    }
    mismatches = {
        key: {"attestation": attestation.get(key), "manifest": value}
        for key, value in expected_attestation.items()
        if attestation.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Runtime attestation does not match the manifest: {mismatches}")
    current_stats = checkpoint_file_stats(runtime_model_path)
    if current_stats != attestation.get("checkpoint_file_stats"):
        raise RuntimeError("Checkpoint files changed after full runtime attestation")


def make_runtime_attestation(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matrix_config = Path(args.matrix_config or manifest["matrix_config"]).resolve()
    validate_manifest_runtime_contract(
        manifest,
        args,
        matrix_config,
        require_attestation=False,
    )
    runtime_model_path = args.model_path or manifest["model_identity"]["resolved_path"]
    current_identity = checkpoint_identity(runtime_model_path)
    if current_identity != manifest["model_identity"]:
        raise RuntimeError("Full checkpoint hash differs from the manifest identity")
    payload = {
        "schema": "topic-image-replay/readout-v2-runtime-attestation/v1",
        "created_at": time.time(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "repo_head": manifest["repo_snapshot"]["head"],
        "implementation_sha256": manifest["implementation_sha256"],
        "model_identity_sha256": manifest["model_identity_sha256"],
        "source_data_sha256": manifest["source_data_sha256"],
        "checkpoint_file_stats": checkpoint_file_stats(runtime_model_path),
    }
    write_json(Path(args.output).resolve(), payload)
    return payload


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    from vlmeval.dataset import build_dataset

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest["conditions"] != list(CONDITIONS):
        raise RuntimeError(f"Manifest condition mismatch: {manifest['conditions']}")
    matrix_config = Path(args.matrix_config or manifest["matrix_config"]).resolve()
    if hashlib.sha256(matrix_config.read_bytes()).hexdigest() != manifest["matrix_config_sha256"]:
        raise RuntimeError(f"Matrix config changed after manifest creation: {matrix_config}")
    validate_manifest_runtime_contract(manifest, args, matrix_config)
    provenance = expected_provenance(manifest)
    if args.smoke_one_per_choice_count:
        tasks = []
        for dataset_name in args.datasets:
            dataset_records = [
                row for row in manifest["records"] if row["dataset"] == dataset_name
            ]
            choice_counts = sorted({int(row["choice_count"]) for row in dataset_records})
            if not choice_counts:
                raise RuntimeError(f"No manifest records for smoke dataset {dataset_name}")
            for choice_count in choice_counts:
                tasks.append(
                    next(
                        row
                        for row in dataset_records
                        if int(row["choice_count"]) == choice_count
                    )
                )
    elif args.smoke_one_per_dataset:
        tasks = []
        for dataset_name in args.datasets:
            candidates = [row for row in manifest["records"] if row["dataset"] == dataset_name]
            if not candidates:
                raise RuntimeError(f"No manifest records for smoke dataset {dataset_name}")
            tasks.append(candidates[0])
    else:
        tasks = [
            row
            for row in manifest["records"]
            if row["dataset"] in args.datasets and int(row["shard"]) == int(args.shard_rank)
        ]
    output_jsonl = Path(args.output_jsonl).resolve()
    resume_shard_rank = (
        None
        if args.smoke_one_per_dataset or args.smoke_one_per_choice_count
        else int(args.shard_rank)
    )
    completed = (
        load_completed(
            output_jsonl,
            manifest,
            provenance,
            expected_shard_rank=resume_shard_rank,
        )
        if args.resume
        else set()
    )
    tasks = [row for row in tasks if (row["dataset"], row["sample_index"]) not in completed]
    if not tasks:
        return {"processed": 0, "skipped_completed": len(completed)}

    repo_root = Path(args.repo_root).resolve()
    first_dataset = tasks[0]["dataset"]
    first_env, first_runner, _ = build_runtime(
        repo_root,
        Path(args.runtime_root).resolve(),
        first_dataset,
        args.model_key,
        args.gpu_id,
        args.model_path,
        args.lmu_data,
        str(matrix_config),
    )
    registry_name = first_runner.models[args.model_key].registry_name
    if registry_name != manifest.get("registry_name"):
        raise RuntimeError(
            f"Model registry changed after manifest creation: "
            f"{registry_name} != {manifest.get('registry_name')}"
        )
    wrapper = load_qwen_model(first_env, registry_name)
    processed = 0
    by_dataset = {name: [row for row in tasks if row["dataset"] == name] for name in args.datasets}
    for dataset_name, dataset_tasks in by_dataset.items():
        if not dataset_tasks:
            continue
        env, runner, _ = build_runtime(
            repo_root,
            Path(args.runtime_root).resolve(),
            dataset_name,
            args.model_key,
            args.gpu_id,
            args.model_path,
            args.lmu_data,
            str(matrix_config),
        )
        if runner.models[args.model_key].registry_name != registry_name:
            raise RuntimeError("One shard cannot switch model registries")
        refresh_replay_runtime(wrapper, env)
        with patched_environ(env):
            dataset = build_dataset(dataset_name)
            for local_position, task in enumerate(dataset_tasks):
                row = dataset.data.iloc[int(task["row_position"])]
                if str(row.get("index")) != task["sample_index"]:
                    raise RuntimeError(f"Manifest/data index mismatch: {task} vs {row.get('index')}")
                current_records, _ = selected_subset_records(
                    dataset_name,
                    row.to_frame().T,
                    selection_profile=manifest.get("selection_profile", "fixed_choice"),
                )
                if len(current_records) != 1:
                    raise RuntimeError(
                        f"Manifest row no longer satisfies selection contract: {task}"
                    )
                validate_semantic_match(
                    task,
                    current_records[0],
                    context=f"runtime manifest key={record_key(task)}",
                )
                dump_dir = None
                diagnostics = bool(args.diagnostics)
                if args.dump_raw_root:
                    dump_dir = (
                        Path(args.dump_raw_root).resolve()
                        / dataset_name
                        / f"row_{task['row_position']}_index_{task['sample_index']}"
                    )
                record = score_record(
                    wrapper,
                    dataset,
                    dataset_name,
                    row,
                    task,
                    dump_dir=dump_dir,
                    diagnostics=diagnostics,
                )
                record["runtime"] = {
                    "model_key": args.model_key,
                    "registry_name": registry_name,
                    "runtime_env": env_subset(env),
                    "shard_rank": int(args.shard_rank),
                    "gpu_id": args.gpu_id,
                }
                record["provenance"] = provenance
                if dump_dir is not None:
                    bind_dump_artifact(dump_dir, record, task, manifest)
                append_jsonl(output_jsonl, record)
                processed += 1
                print(
                    json.dumps(
                        {
                            "dataset": dataset_name,
                            "sample_index": task["sample_index"],
                            "processed": processed,
                            "remaining": len(tasks) - processed,
                            "hits": {name: record["conditions"][name]["hit"] for name in CONDITIONS},
                            "seconds": record["timing_seconds"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return {"processed": processed, "skipped_completed": len(completed)}


REUSE_SEMANTIC_FIELDS = (
    "dataset",
    "row_position",
    "sample_index",
    "answer_key",
    "choice_labels",
    "choice_count",
    "option_text_sha256",
)


def record_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["sample_index"])


def manifest_record_map(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    if sha256_json(manifest["records"]) != manifest["records_sha256"]:
        raise RuntimeError("Manifest records do not match their records hash")
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for row in manifest["records"]:
        key = record_key(row)
        if key in records:
            raise RuntimeError(f"Manifest contains duplicate key: {key}")
        records[key] = row
    return records


def validate_semantic_match(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    context: str,
) -> None:
    mismatches = {
        field: {"expected": expected.get(field), "actual": actual.get(field)}
        for field in REUSE_SEMANTIC_FIELDS
        if expected.get(field) != actual.get(field)
    }
    if mismatches:
        raise RuntimeError(f"Semantic mismatch for {context}: {mismatches}")


def validated_prediction_map(
    manifest: dict[str, Any],
    paths: list[Path],
    *,
    require_complete: bool,
) -> dict[tuple[str, str], dict[str, Any]]:
    if not paths:
        raise RuntimeError("No prediction files found")
    manifest_records = manifest_record_map(manifest)
    expected_keys = set(manifest_records)
    provenance = expected_provenance(manifest)
    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        match = re.fullmatch(r"shard(\d+)\.jsonl", path.name)
        if match is None:
            raise RuntimeError(f"Prediction file has a noncanonical shard name: {path.name}")
        expected_shard_rank = int(match.group(1))
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = record_key(row)
                if key in predictions:
                    raise RuntimeError(f"Prediction files contain duplicate key: {key}")
                if key not in expected_keys:
                    raise RuntimeError(f"Prediction key is absent from its manifest: {key}")
                validate_prediction_payload(
                    row,
                    manifest_records[key],
                    provenance,
                    expected_shard_rank=expected_shard_rank,
                )
                predictions[key] = row
    if require_complete and set(predictions) != expected_keys:
        missing = sorted(expected_keys - set(predictions))[:10]
        raise RuntimeError(
            f"Prediction set is incomplete: {len(predictions)}/{len(expected_keys)}, "
            f"missing={missing}"
        )
    return predictions


def git_text(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args],
        text=True,
    ).strip()


def git_blob(repo_root: Path, revision_path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "show", revision_path],
    )


def resolve_reuse_code_ref(
    repo_root: Path,
    reuse_manifest: dict[str, Any],
    requested_ref: str,
) -> str:
    if requested_ref:
        candidates = [requested_ref]
    else:
        candidates = [str(reuse_manifest["repo_snapshot"]["head"])]
        history = git_text(
            repo_root,
            "rev-list",
            "--all",
            "--",
            "vlmeval/probes/readout_v2.py",
        ).splitlines()
        candidates.extend(history)
    expected_sha256 = reuse_manifest["implementation_sha256"]
    failures = []
    for candidate in dict.fromkeys(candidates):
        try:
            source = git_blob(
                repo_root,
                f"{candidate}:vlmeval/probes/readout_v2.py",
            )
        except subprocess.CalledProcessError:
            failures.append(f"{candidate}:missing")
            continue
        actual_sha256 = hashlib.sha256(source).hexdigest()
        if actual_sha256 == expected_sha256:
            return candidate
        failures.append(f"{candidate}:{actual_sha256}")
    raise RuntimeError(
        "Could not resolve a commit containing the accepted readout implementation: "
        f"{failures}"
    )


def validate_reuse_artifact_lock(
    reuse_manifest_path: Path,
    reuse_manifest: dict[str, Any],
    reuse_paths: list[Path],
    reuse_lock_path: Path,
    expected_lock_sha256: str,
) -> dict[str, Any]:
    actual_lock_sha256 = sha256_file(reuse_lock_path)
    if actual_lock_sha256 != expected_lock_sha256:
        raise RuntimeError(
            f"Accepted reuse lock changed: {actual_lock_sha256} != {expected_lock_sha256}"
        )
    lock = json.loads(reuse_lock_path.read_text(encoding="utf-8"))
    provenance = lock.get("provenance", {})
    manifest_sha256 = sha256_file(reuse_manifest_path)
    if provenance.get("manifest_file_sha256") != manifest_sha256:
        raise RuntimeError("Accepted manifest hash differs from its immutable lock")
    expected_shards = provenance.get("prediction_shard_sha256", {})
    actual_shards = {path.name: sha256_file(path) for path in reuse_paths}
    if actual_shards != expected_shards:
        raise RuntimeError("Accepted prediction shard hashes differ from their immutable lock")
    locked_fields = {
        "implementation_sha256": reuse_manifest["implementation_sha256"],
        "records_sha256": reuse_manifest["records_sha256"],
        "repo_head": reuse_manifest["repo_snapshot"]["head"],
    }
    mismatches = {
        field: {"lock": provenance.get(field), "manifest": value}
        for field, value in locked_fields.items()
        if provenance.get(field) != value
    }
    if mismatches:
        raise RuntimeError(f"Accepted reuse lock/manifest mismatch: {mismatches}")
    return {
        "path": str(reuse_lock_path),
        "sha256": actual_lock_sha256,
        "manifest_file_sha256": manifest_sha256,
        "prediction_shard_sha256": actual_shards,
        "source_tsv_sha256": provenance.get("source_tsv_sha256", {}),
    }


def historical_checkpoint_evidence(
    model_identity: dict[str, Any],
    reuse_manifest: dict[str, Any],
) -> dict[str, Any]:
    model_path = model_identity["resolved_path"]
    stats = checkpoint_file_stats(model_path)
    accepted_created_at = float(reuse_manifest["created_at"])
    latest_mtime = max(item["mtime_ns"] for item in stats) / 1e9
    if latest_mtime >= accepted_created_at:
        raise RuntimeError(
            "Checkpoint files were modified after the accepted manifest was created"
        )
    cache_root = Path(model_path) / ".cache" / "huggingface" / "download"
    cache_content_hashes = set()
    revisions = set()
    if cache_root.is_dir():
        for path in cache_root.iterdir():
            match = re.search(r"\.([0-9a-f]{64})\.incomplete$", path.name)
            if match:
                cache_content_hashes.add(match.group(1))
        for path in cache_root.glob("*.metadata"):
            lines = path.read_text(encoding="utf-8").splitlines()
            if lines:
                revisions.add(lines[0])
    large_weight_hashes = {
        item["sha256"]
        for item in model_identity["files"]
        if item["relative_path"].endswith((".safetensors", ".bin", ".pt", ".pth"))
        and int(item["size"]) > 16 * 1024 * 1024
    }
    if cache_content_hashes and not large_weight_hashes <= cache_content_hashes:
        raise RuntimeError("Current weight hashes do not match the local snapshot cache evidence")
    return {
        "accepted_manifest_created_at": accepted_created_at,
        "latest_checkpoint_mtime": latest_mtime,
        "all_checkpoint_mtimes_predate_accepted_manifest": True,
        "resolved_model_path": model_path,
        "current_model_identity_sha256": model_identity["content_sha256"],
        "large_weight_sha256": sorted(large_weight_hashes),
        "cache_content_sha256": sorted(cache_content_hashes),
        "snapshot_revisions": sorted(revisions),
        "limitation": (
            "The accepted run did not record a checkpoint content hash; path, pre-run mtimes, "
            "snapshot metadata, and current weight hashes are retrospective evidence, not a "
            "cryptographic identity captured at inference time."
        ),
    }


def validate_reuse_scoring_contract(
    all_manifest: dict[str, Any],
    reuse_manifest: dict[str, Any],
    reuse_code_ref: str,
) -> dict[str, Any]:
    repo_root = Path(all_manifest["repo_root"]).resolve()
    status = git_text(repo_root, "status", "--short")
    if status:
        raise RuntimeError(f"Reuse derivation requires a clean repository: {status}")
    old_source_bytes = git_blob(
        repo_root,
        f"{reuse_code_ref}:vlmeval/probes/readout_v2.py",
    )
    old_source = old_source_bytes.decode("utf-8")
    old_source_sha = hashlib.sha256(old_source_bytes).hexdigest()
    if old_source_sha != reuse_manifest["implementation_sha256"]:
        raise RuntimeError(
            "Reuse code ref does not match the accepted manifest implementation: "
            f"{old_source_sha} != {reuse_manifest['implementation_sha256']}"
        )
    old_contract = scoring_contract_sha256_from_source(old_source)
    current_contract = current_scoring_contract_sha256()
    if old_contract != current_contract:
        raise RuntimeError(f"Scoring contract changed: {old_contract} != {current_contract}")
    if all_manifest.get("scoring_contract_sha256") != current_contract:
        raise RuntimeError("All-qualified manifest scoring contract is stale")
    changed_vlmeval_paths = [
        line
        for line in git_text(
            repo_root,
            "diff",
            "--name-only",
            reuse_code_ref,
            "HEAD",
            "--",
            "vlmeval",
        ).splitlines()
        if line
    ]
    allowed_changed_paths = {
        "vlmeval/dataset/dynamath.py",
        "vlmeval/probes/readout_v2.py",
    }
    unexpected = sorted(set(changed_vlmeval_paths) - allowed_changed_paths)
    if unexpected:
        raise RuntimeError(f"Unexpected runtime-code changes since accepted run: {unexpected}")
    from vlmeval.dataset.dynamath import _multiple_choice_instruction

    accepted_dynamath_records = [
        row for row in reuse_manifest["records"] if row["dataset"] == "DynaMath"
    ]
    if any(row["choice_labels"] != list("ABCD") for row in accepted_dynamath_records):
        raise RuntimeError("Accepted DynaMath reuse contains a non-ABCD row")
    four_choice_question = "Question\nA: first\nB: second\nC: third\nD: fourth"
    expected_json_instruction = (
        "Provide the corresponing choice option in the 'short answer' key, "
        "such as 'A', 'B', 'C', or 'D'."
    )
    expected_direct_instruction = (
        "Answer with only the corresponding choice option, such as 'A', 'B', 'C', or 'D'."
    )
    actual_json_instruction = _multiple_choice_instruction(four_choice_question)
    actual_direct_instruction = _multiple_choice_instruction(
        four_choice_question,
        directly_answer=True,
    )
    if actual_json_instruction != expected_json_instruction:
        raise RuntimeError("DynaMath four-choice JSON instruction changed")
    if actual_direct_instruction != expected_direct_instruction:
        raise RuntimeError("DynaMath four-choice direct instruction changed")
    for field in ("model_key", "replay_mode", "policy", "answer_prefix", "conditions"):
        if all_manifest.get(field) != reuse_manifest.get(field):
            raise RuntimeError(
                f"Reuse contract field changed for {field}: "
                f"{all_manifest.get(field)} != {reuse_manifest.get(field)}"
            )
    if all_manifest["matrix_config_sha256"] != reuse_manifest["matrix_config_sha256"]:
        raise RuntimeError("Matrix config changed since the accepted run")
    return {
        "reuse_code_ref": reuse_code_ref,
        "accepted_implementation_sha256": old_source_sha,
        "accepted_scoring_contract_sha256": old_contract,
        "current_scoring_contract_sha256": current_contract,
        "changed_vlmeval_paths": changed_vlmeval_paths,
        "dynamath_four_choice_prompt_compatibility": {
            "accepted_record_count": len(accepted_dynamath_records),
            "json_instruction": actual_json_instruction,
            "direct_instruction": actual_direct_instruction,
        },
    }


def derive_missing_manifest(args: argparse.Namespace) -> dict[str, Any]:
    all_manifest_path = Path(args.all_manifest).resolve()
    reuse_manifest_path = Path(args.reuse_manifest).resolve()
    all_manifest = json.loads(all_manifest_path.read_text(encoding="utf-8"))
    reuse_manifest = json.loads(reuse_manifest_path.read_text(encoding="utf-8"))
    if all_manifest.get("selection_profile") != "all_single_choice":
        raise RuntimeError("Expected an all-single-choice parent manifest")
    reuse_paths = sorted(Path(args.reuse_input_root).resolve().glob(args.reuse_glob))
    reuse_lock_path = Path(args.reuse_lock).resolve()
    artifact_lock = validate_reuse_artifact_lock(
        reuse_manifest_path,
        reuse_manifest,
        reuse_paths,
        reuse_lock_path,
        args.reuse_lock_sha256,
    )
    reuse_predictions = validated_prediction_map(
        reuse_manifest,
        reuse_paths,
        require_complete=True,
    )
    locked_source_hashes = artifact_lock["source_tsv_sha256"]
    for dataset_name in ("DynaMath", "WeMath"):
        if all_manifest["source_data"][dataset_name]["sha256"] != locked_source_hashes.get(
            dataset_name
        ):
            raise RuntimeError(f"Current {dataset_name} source differs from the accepted run")
    accepted_model_paths = {
        row.get("runtime", {}).get("runtime_env", {}).get("MODEL_PATH")
        for row in reuse_predictions.values()
    }
    if accepted_model_paths != {all_manifest["model_identity"]["resolved_path"]}:
        raise RuntimeError(
            f"Accepted prediction model path differs from the current checkpoint: "
            f"{accepted_model_paths}"
        )
    accepted_registry_names = {
        row.get("runtime", {}).get("registry_name") for row in reuse_predictions.values()
    }
    if accepted_registry_names != {all_manifest["registry_name"]}:
        raise RuntimeError(
            f"Accepted prediction registry differs from the current registry: "
            f"{accepted_registry_names}"
        )
    reuse_code_ref = resolve_reuse_code_ref(
        Path(all_manifest["repo_root"]).resolve(),
        reuse_manifest,
        args.reuse_code_ref,
    )
    scoring_validation = validate_reuse_scoring_contract(
        all_manifest,
        reuse_manifest,
        reuse_code_ref,
    )
    checkpoint_evidence = historical_checkpoint_evidence(
        all_manifest["model_identity"],
        reuse_manifest,
    )
    all_records = manifest_record_map(all_manifest)
    reuse_records = manifest_record_map(reuse_manifest)
    reused_keys: set[tuple[str, str]] = set()
    for key, all_record in all_records.items():
        if key not in reuse_records:
            continue
        validate_semantic_match(
            all_record,
            reuse_records[key],
            context=f"reuse manifest key={key}",
        )
        if key not in reuse_predictions:
            raise RuntimeError(f"Accepted prediction missing for reusable key: {key}")
        reused_keys.add(key)

    missing_records = [
        copy.deepcopy(row)
        for row in all_manifest["records"]
        if record_key(row) not in reused_keys
    ]
    dataset_positions: Counter[str] = Counter()
    for row in missing_records:
        dataset_name = row["dataset"]
        position = int(dataset_positions[dataset_name])
        row["dataset_position"] = position
        row["shard"] = position % int(args.num_shards)
        dataset_positions[dataset_name] += 1

    reuse_dataset_counts = Counter(key[0] for key in reused_keys)
    missing_dataset_counts = Counter(row["dataset"] for row in missing_records)
    dataset_summaries = []
    for dataset_meta in all_manifest["datasets"]:
        item = copy.deepcopy(dataset_meta)
        dataset_name = item["dataset"]
        item["all_selected_rows"] = int(item["selected_rows"])
        item["reused_rows"] = int(reuse_dataset_counts[dataset_name])
        item["selected_rows"] = int(missing_dataset_counts[dataset_name])
        dataset_rows = [row for row in missing_records if row["dataset"] == dataset_name]
        item["answer_histogram"] = dict(Counter(row["answer_key"] for row in dataset_rows))
        item["selected_choice_count_histogram"] = dict(
            Counter(row["choice_count"] for row in dataset_rows)
        )
        item["subset"] = "all_single_choice_missing_only"
        dataset_summaries.append(item)

    reused_key_rows = [list(key) for key in sorted(reused_keys)]
    payload = copy.deepcopy(all_manifest)
    payload.update(
        {
            "schema": "topic-image-replay/readout-v2-missing-single-choice/v1",
            "created_at": time.time(),
            "num_shards": int(args.num_shards),
            "datasets": dataset_summaries,
            "record_count": len(missing_records),
            "records_sha256": sha256_json(missing_records),
            "records": missing_records,
            "parent_all_manifest": str(all_manifest_path),
            "parent_all_manifest_sha256": sha256_file(all_manifest_path),
            "parent_all_records_sha256": all_manifest["records_sha256"],
            "reuse": {
                "source_manifest": str(reuse_manifest_path),
                "source_manifest_sha256": sha256_file(reuse_manifest_path),
                "source_records_sha256": reuse_manifest["records_sha256"],
                "source_prediction_files": artifact_lock["prediction_shard_sha256"],
                "artifact_lock": artifact_lock,
                "historical_checkpoint_evidence": checkpoint_evidence,
                "reused_record_count": len(reused_keys),
                "reused_dataset_counts": dict(reuse_dataset_counts),
                "reused_keys_sha256": sha256_json(reused_key_rows),
                "scoring_validation": scoring_validation,
            },
        }
    )
    if len(missing_records) + len(reused_keys) != len(all_records):
        raise RuntimeError("Reuse and missing partitions do not cover the all-qualified manifest")
    write_json(Path(args.output).resolve(), payload)
    return payload


def validate_selection_contract(args: argparse.Namespace) -> dict[str, Any]:
    all_manifest_path = Path(args.all_manifest).resolve()
    missing_manifest_path = Path(args.missing_manifest).resolve()
    all_manifest = json.loads(all_manifest_path.read_text(encoding="utf-8"))
    missing_manifest = json.loads(missing_manifest_path.read_text(encoding="utf-8"))
    manifest_record_map(all_manifest)
    manifest_record_map(missing_manifest)
    if all_manifest.get("selection_profile") != "all_single_choice":
        raise RuntimeError("Selection gate requires an all-single-choice manifest")
    if all_manifest["record_count"] != EXPECTED_ALL_SINGLE_CHOICE_TOTAL:
        raise RuntimeError(
            f"All-qualified count changed: {all_manifest['record_count']} "
            f"!= {EXPECTED_ALL_SINGLE_CHOICE_TOTAL}"
        )
    all_dataset_meta = {item["dataset"]: item for item in all_manifest["datasets"]}
    if set(all_dataset_meta) != set(EXPECTED_ALL_SINGLE_CHOICE_COUNTS):
        raise RuntimeError(f"All-qualified dataset set changed: {sorted(all_dataset_meta)}")
    for dataset_name, expected in EXPECTED_ALL_SINGLE_CHOICE_COUNTS.items():
        actual = all_dataset_meta[dataset_name]
        histogram = {
            int(key): int(value)
            for key, value in actual["selected_choice_count_histogram"].items()
        }
        if int(actual["selected_rows"]) != expected["rows"]:
            raise RuntimeError(f"Eligible row count changed for {dataset_name}")
        if histogram != expected["choice_counts"]:
            raise RuntimeError(f"Choice-count histogram changed for {dataset_name}: {histogram}")

    reuse_counts = {
        key: int(value)
        for key, value in missing_manifest["reuse"]["reused_dataset_counts"].items()
        if int(value)
    }
    if reuse_counts != EXPECTED_REUSE_COUNTS:
        raise RuntimeError(f"Immutable reuse counts changed: {reuse_counts}")
    if int(missing_manifest["reuse"]["reused_record_count"]) != sum(
        EXPECTED_REUSE_COUNTS.values()
    ):
        raise RuntimeError("Immutable reuse total changed")
    if missing_manifest["record_count"] != EXPECTED_MISSING_SINGLE_CHOICE_TOTAL:
        raise RuntimeError(
            f"Missing-only count changed: {missing_manifest['record_count']} "
            f"!= {EXPECTED_MISSING_SINGLE_CHOICE_TOTAL}"
        )
    if (
        int(missing_manifest["record_count"])
        + int(missing_manifest["reuse"]["reused_record_count"])
        != int(all_manifest["record_count"])
    ):
        raise RuntimeError("Selection gate partition totals do not cover the all manifest")
    payload = {
        "schema": "topic-image-replay/readout-v2-selection-validation/v1",
        "pass": True,
        "all_manifest": str(all_manifest_path),
        "all_manifest_sha256": sha256_file(all_manifest_path),
        "all_record_count": all_manifest["record_count"],
        "missing_manifest": str(missing_manifest_path),
        "missing_manifest_sha256": sha256_file(missing_manifest_path),
        "missing_record_count": missing_manifest["record_count"],
        "reused_record_count": missing_manifest["reuse"]["reused_record_count"],
        "datasets": all_manifest["datasets"],
    }
    write_json(Path(args.output).resolve(), payload)
    return payload


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    paths = sorted(Path(args.input_root).glob(args.glob))
    by_key = validated_prediction_map(manifest, paths, require_complete=False)
    expected = {(row["dataset"], row["sample_index"]) for row in manifest["records"]}
    actual = set(by_key)
    summaries = []
    csv_lines = ["dataset,subset,n,baseline_acc,readout_v2_acc,full_acc"]
    for dataset_meta in manifest["datasets"]:
        dataset_name = dataset_meta["dataset"]
        dataset_rows = [row for (name, _), row in by_key.items() if name == dataset_name]
        item: dict[str, Any] = {
            "dataset": dataset_name,
            "subset": dataset_meta["subset"],
            "n": len(dataset_rows),
        }
        for condition in CONDITIONS:
            item[f"{condition}_acc"] = (
                sum(bool(row["conditions"][condition]["hit"]) for row in dataset_rows) / len(dataset_rows)
                if dataset_rows
                else None
            )
        summaries.append(item)
        csv_lines.append(
            ",".join(
                [
                    dataset_name,
                    dataset_meta["subset"],
                    str(item["n"]),
                    *("" if item[f"{condition}_acc"] is None else f"{item[f'{condition}_acc']:.8f}" for condition in CONDITIONS),
                ]
            )
        )
    payload = {
        "schema": "topic-image-replay/readout-v2-summary/v1",
        "input_files": [str(path) for path in paths],
        "expected_records": len(expected),
        "actual_unique_records": len(actual),
        "missing_records": sorted([list(key) for key in expected - actual]),
        "unexpected_records": sorted([list(key) for key in actual - expected]),
        "duplicate_keys": [],
        "provenance_mismatches": [],
        "complete": actual == expected,
        "datasets": summaries,
    }
    output_root = Path(args.output_root)
    write_json(output_root / "summary.json", payload)
    (output_root / "accuracy.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    if args.require_complete and not payload["complete"]:
        raise SystemExit(2)
    return payload


def aggregate_combined(args: argparse.Namespace) -> dict[str, Any]:
    all_manifest_path = Path(args.all_manifest).resolve()
    reuse_manifest_path = Path(args.reuse_manifest).resolve()
    missing_manifest_path = Path(args.missing_manifest).resolve()
    all_manifest = json.loads(all_manifest_path.read_text(encoding="utf-8"))
    reuse_manifest = json.loads(reuse_manifest_path.read_text(encoding="utf-8"))
    missing_manifest = json.loads(missing_manifest_path.read_text(encoding="utf-8"))
    if missing_manifest.get("parent_all_records_sha256") != all_manifest["records_sha256"]:
        raise RuntimeError("Missing manifest does not derive from the supplied all manifest")
    if missing_manifest.get("parent_all_manifest_sha256") != sha256_file(all_manifest_path):
        raise RuntimeError("All-manifest file hash differs from the missing-manifest contract")
    if missing_manifest["reuse"]["source_manifest_sha256"] != sha256_file(reuse_manifest_path):
        raise RuntimeError("Reuse-manifest file hash differs from the missing-manifest contract")

    reuse_paths = sorted(Path(args.reuse_input_root).resolve().glob(args.reuse_glob))
    missing_paths = sorted(Path(args.missing_input_root).resolve().glob(args.missing_glob))
    actual_reuse_hashes = {path.name: sha256_file(path) for path in reuse_paths}
    if actual_reuse_hashes != missing_manifest["reuse"]["source_prediction_files"]:
        raise RuntimeError("Accepted prediction files changed after missing-manifest derivation")
    artifact_lock = missing_manifest["reuse"]["artifact_lock"]
    artifact_lock_path = Path(args.reuse_lock).resolve()
    if (
        not artifact_lock_path.is_file()
        or sha256_file(artifact_lock_path) != artifact_lock["sha256"]
    ):
        raise RuntimeError("Accepted artifact lock changed after missing-manifest derivation")
    reuse_predictions = validated_prediction_map(
        reuse_manifest,
        reuse_paths,
        require_complete=True,
    )
    missing_predictions = validated_prediction_map(
        missing_manifest,
        missing_paths,
        require_complete=True,
    )
    all_records = manifest_record_map(all_manifest)
    reuse_records = manifest_record_map(reuse_manifest)
    missing_records = manifest_record_map(missing_manifest)
    all_keys = set(all_records)
    reuse_keys = set(reuse_records)
    missing_keys = set(missing_records)
    if not missing_keys <= all_keys:
        raise RuntimeError("Missing manifest contains keys outside the all-qualified manifest")
    for key in missing_keys:
        validate_semantic_match(
            all_records[key],
            missing_records[key],
            context=f"combined missing key={key}",
        )
    eligible_reused_keys: set[tuple[str, str]] = set()
    for key in all_keys & reuse_keys:
        validate_semantic_match(
            all_records[key],
            reuse_records[key],
            context=f"combined eligible-reuse key={key}",
        )
        eligible_reused_keys.add(key)
    expected_missing_keys = all_keys - eligible_reused_keys
    if missing_keys != expected_missing_keys:
        raise RuntimeError(
            "Missing manifest is not the exact complement of semantically eligible reuse: "
            f"missing_only={len(missing_keys - expected_missing_keys)}, "
            f"omitted={len(expected_missing_keys - missing_keys)}"
        )
    expected_reused_keys = eligible_reused_keys
    if len(expected_reused_keys) != int(missing_manifest["reuse"]["reused_record_count"]):
        raise RuntimeError("Reused record count differs from the missing-manifest contract")
    reused_key_hash = sha256_json([list(key) for key in sorted(expected_reused_keys)])
    if reused_key_hash != missing_manifest["reuse"]["reused_keys_sha256"]:
        raise RuntimeError("Reused key hash differs from the missing-manifest contract")

    combined: dict[tuple[str, str], dict[str, Any]] = {}
    for key in sorted(expected_reused_keys):
        if key not in reuse_records or key not in reuse_predictions:
            raise RuntimeError(f"Reusable key lacks accepted evidence: {key}")
        validate_semantic_match(
            all_records[key],
            reuse_records[key],
            context=f"combined reuse key={key}",
        )
        combined[key] = reuse_predictions[key]
    for key, prediction in missing_predictions.items():
        if key not in missing_keys:
            raise RuntimeError(f"Missing prediction is outside the missing manifest: {key}")
        if key in combined:
            raise RuntimeError(f"Combined predictions contain duplicate key: {key}")
        combined[key] = prediction

    actual_keys = set(combined)
    complete = actual_keys == all_keys
    dataset_summaries = []
    csv_lines = ["dataset,subset,n,baseline_acc,readout_v2_acc,full_acc"]
    for dataset_meta in all_manifest["datasets"]:
        dataset_name = dataset_meta["dataset"]
        dataset_rows = [
            row for (name, _), row in combined.items() if name == dataset_name
        ]
        item: dict[str, Any] = {
            "dataset": dataset_name,
            "subset": dataset_meta["subset"],
            "n": len(dataset_rows),
            "reused_n": sum(key[0] == dataset_name for key in expected_reused_keys),
            "new_n": sum(key[0] == dataset_name for key in missing_keys),
        }
        for condition in CONDITIONS:
            item[f"{condition}_acc"] = (
                sum(bool(row["conditions"][condition]["hit"]) for row in dataset_rows)
                / len(dataset_rows)
                if dataset_rows
                else None
            )
        dataset_summaries.append(item)
        csv_lines.append(
            ",".join(
                [
                    dataset_name,
                    dataset_meta["subset"],
                    str(item["n"]),
                    *(
                        ""
                        if item[f"{condition}_acc"] is None
                        else f"{item[f'{condition}_acc']:.8f}"
                        for condition in CONDITIONS
                    ),
                ]
            )
        )

    payload = {
        "schema": "topic-image-replay/readout-v2-combined-summary/v1",
        "complete": complete,
        "all_manifest": str(all_manifest_path),
        "all_manifest_sha256": sha256_file(all_manifest_path),
        "reuse_manifest": str(reuse_manifest_path),
        "reuse_manifest_sha256": sha256_file(reuse_manifest_path),
        "missing_manifest": str(missing_manifest_path),
        "missing_manifest_sha256": sha256_file(missing_manifest_path),
        "reuse_prediction_files": {
            path.name: sha256_file(path) for path in reuse_paths
        },
        "missing_prediction_files": {
            path.name: sha256_file(path) for path in missing_paths
        },
        "expected_records": len(all_keys),
        "actual_unique_records": len(actual_keys),
        "reused_records": len(expected_reused_keys),
        "new_records": len(missing_keys),
        "missing_records": sorted([list(key) for key in all_keys - actual_keys]),
        "unexpected_records": sorted([list(key) for key in actual_keys - all_keys]),
        "datasets": dataset_summaries,
    }
    output_root = Path(args.output_root).resolve()
    write_json(output_root / "summary.json", payload)
    (output_root / "accuracy.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    if args.require_complete and not complete:
        raise SystemExit(2)
    return payload


def independent_expected_masks(
    seq_len: int,
    decode_start: int,
    i2_start: int,
    i2_end: int,
) -> np.ndarray:
    expected = np.zeros((len(CONDITIONS), seq_len, seq_len), dtype=np.bool_)
    for query in range(seq_len):
        if query < decode_start:
            expected[:, query, : query + 1] = True
            continue
        expected[0, query, decode_start : query + 1] = True
        expected[1, query, decode_start : query + 1] = True
        expected[1, query, i2_start : i2_end + 1] = True
        expected[2, query, : query + 1] = True
    return expected


def raw_contiguous_spans(values: np.ndarray, target: int) -> list[tuple[int, int]]:
    positions = np.flatnonzero(values == target).tolist()
    if not positions:
        return []
    spans = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            spans.append((start, previous))
            start = position
        previous = position
    spans.append((start, previous))
    return spans


def validate_smoke(args: argparse.Namespace) -> dict[str, Any]:
    from vlmeval.dataset.dynamath import _format_choice_option_examples

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_records = manifest_record_map(manifest)
    provenance = expected_provenance(manifest)
    smoke_jsonl_path = Path(args.smoke_jsonl).resolve()
    smoke_rows: dict[tuple[str, str], dict[str, Any]] = {}
    with smoke_jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record_key(record)
            if key in smoke_rows or key not in manifest_records:
                raise RuntimeError(f"Invalid or duplicate smoke JSONL key: {key}")
            validate_prediction_payload(
                record,
                manifest_records[key],
                provenance,
                expected_shard_rank=None,
            )
            smoke_rows[key] = record

    reuse_comparisons = []
    reuse_overlap_pass = True
    if args.reuse_reference_manifest:
        reuse_manifest = json.loads(
            Path(args.reuse_reference_manifest).read_text(encoding="utf-8")
        )
        reuse_paths = sorted(
            Path(args.reuse_reference_root).resolve().glob(args.reuse_reference_glob)
        )
        reuse_predictions = validated_prediction_map(
            reuse_manifest,
            reuse_paths,
            require_complete=True,
        )
        reuse_records = manifest_record_map(reuse_manifest)
        overlap = sorted(set(smoke_rows) & set(reuse_predictions))
        for key in overlap:
            validate_semantic_match(
                manifest_records[key],
                reuse_records[key],
                context=f"smoke reuse sentinel key={key}",
            )
            condition_diffs = {}
            predicted_equal = True
            for condition in CONDITIONS:
                current = smoke_rows[key]["conditions"][condition]
                accepted = reuse_predictions[key]["conditions"][condition]
                labels = manifest_records[key]["choice_labels"]
                condition_diffs[condition] = max(
                    abs(
                        float(current["candidate_logprobs"][label])
                        - float(accepted["candidate_logprobs"][label])
                    )
                    for label in labels
                )
                predicted_equal = predicted_equal and (
                    current["predicted_key"] == accepted["predicted_key"]
                )
            row_pass = predicted_equal and max(condition_diffs.values()) <= 0.25
            reuse_overlap_pass = reuse_overlap_pass and row_pass
            reuse_comparisons.append(
                {
                    "dataset": key[0],
                    "sample_index": key[1],
                    "candidate_logprob_max_abs_diff": condition_diffs,
                    "predicted_keys_equal": predicted_equal,
                    "pass": row_pass,
                }
            )
        reuse_overlap_pass = reuse_overlap_pass and (
            len(overlap) == int(args.expected_reuse_overlap)
        )

    root = Path(args.raw_root).resolve()
    artifacts = sorted(root.glob("**/artifact.json"))
    rows = []
    artifact_keys = set()
    for artifact_path in artifacts:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        key = record_key(artifact)
        if key in artifact_keys or key not in smoke_rows:
            raise RuntimeError(f"Invalid or duplicate raw artifact key: {key}")
        artifact_keys.add(key)
        smoke_record = smoke_rows[key]
        manifest_record = manifest_records[key]
        raw_dir = artifact_path.parent
        input_ids = np.load(raw_dir / artifact["raw_files"]["input_ids"])
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise RuntimeError(f"Unexpected raw input_ids shape for {key}: {input_ids.shape}")
        ids = input_ids[0]
        seq_len = int(ids.shape[0])
        decode_start = int(artifact["prefill_token_count"])
        token_table = artifact["token_table"]
        token_table_matches_raw = len(token_table) == seq_len and all(
            int(item["position"]) == position and int(item["token_id"]) == int(ids[position])
            for position, item in enumerate(token_table)
        )
        image_token_ids = {
            int(item["token_id"])
            for item in token_table
            if item.get("token") == "<|image_pad|>"
        }
        if len(image_token_ids) != 1:
            raise RuntimeError(f"Could not identify one image-pad token id for {key}")
        image_spans = raw_contiguous_spans(ids, next(iter(image_token_ids)))
        if len(image_spans) != 2:
            raise RuntimeError(f"Raw input does not contain exactly two image cores for {key}")
        i2_start, i2_end = image_spans[1]
        expected_masks = independent_expected_masks(
            seq_len,
            decode_start,
            i2_start,
            i2_end,
        )
        allowed = np.load(raw_dir / artifact["raw_files"]["allowed_masks"])
        plan = artifact["candidate_plan"]
        labels = list(artifact["choice_labels"])
        forced_prefix_ids = list(plan["forced_prefix_ids"])
        seed_ids = list(artifact["boundary_meta"]["assistant_seed_token_ids"])
        expected_tail = seed_ids + forced_prefix_ids
        suffixes = {
            label: plan["joint_token_ids"][label][len(forced_prefix_ids) :]
            for label in labels
        }
        batch_logits = np.load(raw_dir / artifact["raw_files"]["last_token_logits"])
        single_logits_path = artifact["raw_files"].get("last_token_logits_single4d")
        single_logits = np.load(raw_dir / single_logits_path) if single_logits_path else None
        candidate_ids = [int(plan["candidate_token_ids"][label]) for label in labels]
        batch_single_diffs = (
            np.abs(batch_logits[:, candidate_ids] - single_logits[:, candidate_ids])
            if single_logits is not None
            else None
        )
        raw_predictions = {}
        raw_logprob_match = True
        for condition_index, condition in enumerate(CONDITIONS):
            logits = batch_logits[condition_index].astype(np.float64)
            maximum = float(np.max(logits))
            logsumexp = maximum + math.log(float(np.exp(logits - maximum).sum()))
            raw_values = {
                label: float(logits[token_id] - logsumexp)
                for label, token_id in zip(labels, candidate_ids)
            }
            raw_predictions[condition] = max(labels, key=lambda label: raw_values[label])
            stored_values = smoke_record["conditions"][condition]["candidate_logprobs"]
            raw_logprob_match = raw_logprob_match and all(
                math.isclose(raw_values[label], float(stored_values[label]), abs_tol=1e-4)
                for label in labels
            )
        dynamath_prompt_matches_choice_labels = True
        if artifact["dataset"] == "DynaMath":
            choice_examples = _format_choice_option_examples(labels)
            expected_instruction = (
                "Answer with only the corresponding choice option, "
                f"such as {choice_examples}."
            )
            standard_text = "\n".join(
                str(item.get("value", ""))
                for item in artifact["standard_prompt"]
                if item.get("type") == "text"
            )
            dynamath_prompt_matches_choice_labels = expected_instruction in standard_text
        binding = artifact.get("manifest_binding", {})
        artifact_record_fields_match = all(
            artifact.get(field) == smoke_record.get(field)
            for field in (
                "dataset",
                "row_position",
                "sample_index",
                "answer_key",
                "choice_labels",
                "conditions",
                "mask_checks",
                "runtime",
                "provenance",
            )
        )
        checks = {
            "artifact_jsonl_record_exact": artifact_record_fields_match,
            "artifact_jsonl_hash_exact": artifact.get("jsonl_record_sha256")
            == sha256_json(smoke_record),
            "artifact_manifest_record_exact": binding.get("manifest_record_sha256")
            == sha256_json(manifest_record),
            "artifact_manifest_provenance_exact": (
                binding.get("records_sha256") == manifest["records_sha256"]
                and binding.get("model_identity_sha256")
                == manifest.get("model_identity_sha256")
                and binding.get("source_data_sha256") == manifest.get("source_data_sha256")
            ),
            "raw_input_length_exact": seq_len == int(artifact["token_count"]),
            "token_table_matches_raw_input_ids": token_table_matches_raw,
            "raw_i2_derived_independently": (
                int(artifact["readout_v2_span"]["start"]) == i2_start
                and int(artifact["readout_v2_span"]["end"]) == i2_end
                and artifact["readout_v2_span"].get("span_kind") == "image2_visual_core"
            ),
            "raw_mask_exact_independent": bool(np.array_equal(allowed, expected_masks)),
            "content_is_iqi": artifact.get("content_counts") == {"image": 2, "text": 1},
            "forced_prefix_is_exactly_answer_colon": plan["forced_prefix_text"] == "Answer:",
            "candidate_suffixes_are_exactly_one_token": all(
                len(suffixes[label]) == 1
                and suffixes[label][0] == plan["candidate_token_ids"][label]
                for label in labels
            ),
            "decode_tail_exact": ids[decode_start:].tolist() == expected_tail,
            "raw_logits_reproduce_candidate_logprobs": raw_logprob_match,
            "raw_logits_reproduce_predictions": all(
                raw_predictions[condition]
                == smoke_record["conditions"][condition]["predicted_key"]
                for condition in CONDITIONS
            ),
            "batch3_single4d_candidate_logits_close": batch_single_diffs is not None
            and bool(np.max(batch_single_diffs) <= BATCH_SINGLE_CANDIDATE_ATOL),
            "batch3_single4d_candidate_argmax_equal": single_logits is not None
            and all(
                int(np.argmax(batch_logits[index, candidate_ids]))
                == int(np.argmax(single_logits[index, candidate_ids]))
                for index in range(len(CONDITIONS))
            ),
            "full_4d_standard_2d_parity": bool(
                artifact["diagnostics"].get("full_4d_vs_standard_2d_candidate_argmax_equal")
            ),
            "blocked_corruption_invariant": bool(
                artifact["diagnostics"].get("baseline_blocked_corruption_pass_1e_5")
            ),
            "dynamath_prompt_matches_choice_labels": dynamath_prompt_matches_choice_labels,
        }
        checks["pass"] = all(checks.values())
        rows.append(
            {
                "artifact": str(artifact_path),
                "dataset": artifact["dataset"],
                "sample_index": artifact["sample_index"],
                "choice_count": len(labels),
                "checks": checks,
            }
        )
    if artifact_keys != set(smoke_rows):
        raise RuntimeError("Raw artifact keys do not exactly match smoke JSONL keys")
    datasets = {row["dataset"] for row in rows}
    actual_strata = {(row["dataset"], row["choice_count"]) for row in rows}
    expected_strata = {
        (row["dataset"], int(row["choice_count"]))
        for row in manifest["records"]
        if row["dataset"] in args.datasets
    }
    payload = {
        "schema": "topic-image-replay/readout-v2-smoke-validation/v2",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "smoke_jsonl": str(smoke_jsonl_path),
        "smoke_jsonl_sha256": sha256_file(smoke_jsonl_path),
        "artifact_count": len(rows),
        "datasets": sorted(datasets),
        "expected_datasets": list(args.datasets),
        "all_datasets_present": datasets == set(args.datasets),
        "actual_choice_count_strata": [list(item) for item in sorted(actual_strata)],
        "expected_choice_count_strata": [list(item) for item in sorted(expected_strata)],
        "all_choice_count_strata_present": actual_strata == expected_strata,
        "exactly_one_artifact_per_stratum": len(rows) == len(expected_strata),
        "reuse_reference_overlap_expected": int(args.expected_reuse_overlap),
        "reuse_reference_overlap_actual": len(reuse_comparisons),
        "reuse_reference_comparisons": reuse_comparisons,
        "reuse_reference_overlap_pass": reuse_overlap_pass,
        "rows": rows,
    }
    payload["all_pass"] = (
        payload["all_datasets_present"]
        and payload["all_choice_count_strata_present"]
        and payload["exactly_one_artifact_per_stratum"]
        and payload["reuse_reference_overlap_pass"]
        and all(row["checks"]["pass"] for row in rows)
    )
    write_json(Path(args.output), payload)
    if not payload["all_pass"]:
        raise SystemExit(2)
    return payload


def parse_datasets(raw: str) -> list[str]:
    values = [item for item in raw.replace(",", " ").split() if item]
    unsupported = [item for item in values if item not in DATASET_TARGETS]
    if unsupported:
        raise ValueError(f"Unsupported datasets: {unsupported}")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-choice IQI readout-v2 probe")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--repo-root", default=str(Path.cwd()))
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--datasets", default="")
    manifest.add_argument(
        "--selection-profile",
        choices=SELECTION_PROFILES,
        default="fixed_choice",
    )
    manifest.add_argument("--model-key", default="qwen25vl_3b")
    manifest.add_argument("--model-path", default="")
    manifest.add_argument("--lmu-data", default=os.environ.get("LMUData", ""))
    manifest.add_argument("--matrix-config", default="")
    manifest.add_argument("--gpu-id", default="0")
    manifest.add_argument("--num-shards", type=int, default=8)

    run = sub.add_parser("run")
    run.add_argument("--repo-root", default=str(Path.cwd()))
    run.add_argument("--manifest", required=True)
    run.add_argument("--output-jsonl", required=True)
    run.add_argument("--runtime-root", required=True)
    run.add_argument("--datasets", default="")
    run.add_argument("--model-key", default="qwen25vl_3b")
    run.add_argument("--model-path", default="")
    run.add_argument("--lmu-data", default=os.environ.get("LMUData", ""))
    run.add_argument("--matrix-config", default="")
    run.add_argument("--runtime-attestation", required=True)
    run.add_argument("--gpu-id", default="0")
    run.add_argument("--shard-rank", type=int, default=0)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--smoke-one-per-dataset", action="store_true")
    run.add_argument("--smoke-one-per-choice-count", action="store_true")
    run.add_argument("--dump-raw-root", default="")
    run.add_argument("--diagnostics", action="store_true")

    attest = sub.add_parser("attest-runtime")
    attest.add_argument("--repo-root", default=str(Path.cwd()))
    attest.add_argument("--manifest", required=True)
    attest.add_argument("--output", required=True)
    attest.add_argument("--datasets", default="")
    attest.add_argument("--model-key", default="qwen25vl_3b")
    attest.add_argument("--model-path", default="")
    attest.add_argument("--lmu-data", default=os.environ.get("LMUData", ""))
    attest.add_argument("--matrix-config", default="")

    derive = sub.add_parser("derive-missing")
    derive.add_argument("--all-manifest", required=True)
    derive.add_argument("--reuse-manifest", required=True)
    derive.add_argument("--reuse-input-root", required=True)
    derive.add_argument("--reuse-glob", default="shard*.jsonl")
    derive.add_argument("--reuse-lock", required=True)
    derive.add_argument("--reuse-lock-sha256", required=True)
    derive.add_argument("--reuse-code-ref", default="")
    derive.add_argument("--output", required=True)
    derive.add_argument("--num-shards", type=int, default=8)

    selection = sub.add_parser("validate-selection")
    selection.add_argument("--all-manifest", required=True)
    selection.add_argument("--missing-manifest", required=True)
    selection.add_argument("--output", required=True)

    summary = sub.add_parser("aggregate")
    summary.add_argument("--manifest", required=True)
    summary.add_argument("--input-root", required=True)
    summary.add_argument("--glob", default="shard*.jsonl")
    summary.add_argument("--output-root", required=True)
    summary.add_argument("--require-complete", action="store_true")

    combined = sub.add_parser("aggregate-combined")
    combined.add_argument("--all-manifest", required=True)
    combined.add_argument("--reuse-manifest", required=True)
    combined.add_argument("--missing-manifest", required=True)
    combined.add_argument("--reuse-input-root", required=True)
    combined.add_argument("--reuse-glob", default="shard*.jsonl")
    combined.add_argument("--reuse-lock", required=True)
    combined.add_argument("--missing-input-root", required=True)
    combined.add_argument("--missing-glob", default="shard*.jsonl")
    combined.add_argument("--output-root", required=True)
    combined.add_argument("--require-complete", action="store_true")

    validate = sub.add_parser("validate-smoke")
    validate.add_argument("--raw-root", required=True)
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--smoke-jsonl", required=True)
    validate.add_argument("--reuse-reference-manifest", default="")
    validate.add_argument("--reuse-reference-root", default="")
    validate.add_argument("--reuse-reference-glob", default="shard*.jsonl")
    validate.add_argument("--expected-reuse-overlap", type=int, default=0)
    validate.add_argument("--output", required=True)
    validate.add_argument("--datasets", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "datasets"):
        if args.datasets:
            args.datasets = parse_datasets(args.datasets)
        elif args.command == "manifest":
            args.datasets = list(
                ALL_SINGLE_CHOICE_DATASETS
                if args.selection_profile == "all_single_choice"
                else DEFAULT_FIXED_DATASETS
            )
        elif args.command == "run":
            manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            args.datasets = [item["dataset"] for item in manifest["datasets"]]
        elif args.manifest:
            manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            args.datasets = [item["dataset"] for item in manifest["datasets"]]
        else:
            args.datasets = list(DEFAULT_FIXED_DATASETS)
    if args.command == "manifest":
        payload = make_manifest(args)
    elif args.command == "run":
        payload = run_shard(args)
    elif args.command == "attest-runtime":
        payload = make_runtime_attestation(args)
    elif args.command == "derive-missing":
        payload = derive_missing_manifest(args)
    elif args.command == "validate-selection":
        payload = validate_selection_contract(args)
    elif args.command == "aggregate":
        payload = aggregate(args)
    elif args.command == "aggregate-combined":
        payload = aggregate_combined(args)
    else:
        payload = validate_smoke(args)
    printable = payload
    if args.command in {"manifest", "derive-missing"}:
        printable = {
            "schema": payload["schema"],
            "record_count": payload["record_count"],
            "records_sha256": payload["records_sha256"],
            "datasets": payload["datasets"],
        }
        if args.command == "derive-missing":
            printable["reuse"] = payload["reuse"]
    print(json.dumps(printable, ensure_ascii=False, default=json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
