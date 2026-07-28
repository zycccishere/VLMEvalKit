from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
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
    "DynaMath": {"choice_count": 4, "subset": "four_choice"},
    "WeMath": {"choice_count": 5, "subset": "five_choice"},
    "MMBench_DEV_EN_V11": {"choice_count": 4, "subset": "canonical_four_choice"},
}
CONDITIONS = ("baseline", "readout_v2", "full")
REPLAY_MODE = "image_text_image"
POLICY = "direct"
ANSWER_PREFIX = "Answer: "
DEFAULT_MATRIX_CONFIG = "configs/matrix.yaml"
BATCH_SINGLE_CANDIDATE_ATOL = 0.25


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


def embedded_choice_labels(question: str) -> list[str]:
    text = str(question or "")
    found = set(re.findall(r"(?:^|\n)\s*\(?([A-Z])\)?[\.)\:]\s*", text))
    if not found:
        found = set(re.findall(r"\(([A-Z])\)", text))
    labels = []
    for code in range(ord("A"), ord("Z") + 1):
        label = chr(code)
        if label not in found:
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


def selected_subset_records(dataset_name: str, data: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if dataset_name not in DATASET_TARGETS:
        raise KeyError(f"Unsupported fixed-choice dataset: {dataset_name}")
    target = int(DATASET_TARGETS[dataset_name]["choice_count"])
    mmbench_group_sizes: Counter[int] = Counter()
    if dataset_name == "MMBench_DEV_EN_V11":
        mmbench_group_sizes.update(int(row["index"]) % 1_000_000 for _, row in data.iterrows())

    selected: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    for row_position, row in data.iterrows():
        labels = row_choice_labels(dataset_name, row)
        answer = normalize_answer(row.get("answer"))
        if dataset_name == "DynaMath" and str(row.get("answer_type", "")) != "multiple choice":
            rejection_counts["not_multiple_choice"] += 1
            continue
        if labels != list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:target]):
            rejection_counts[f"not_{target}_consecutive_choices"] += 1
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
        if dataset_name != "DynaMath":
            normalized_options = [normalized_option_text(options[label]) for label in labels]
            if any(not value for value in normalized_options):
                rejection_counts["empty_option_text"] += 1
                continue
            if len(set(normalized_options)) != len(normalized_options):
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
        "subset": DATASET_TARGETS[dataset_name]["subset"],
        "target_choice_count": target,
        "total_rows": int(len(data)),
        "selected_rows": len(selected),
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
    runtime_root = output.parent / "_manifest_runtime"
    records: list[dict[str, Any]] = []
    dataset_summaries = []
    for dataset_name in args.datasets:
        env, _, _ = build_runtime(
            repo_root,
            runtime_root,
            dataset_name,
            args.model_key,
            args.gpu_id,
            args.model_path,
            args.lmu_data,
            str(matrix_config),
        )
        with patched_environ(env):
            dataset = build_dataset(dataset_name)
            selected, summary = selected_subset_records(dataset_name, dataset.data)
        for position, item in enumerate(selected):
            item["dataset_position"] = position
            item["shard"] = position % int(args.num_shards)
        records.extend(selected)
        dataset_summaries.append(summary)

    payload = {
        "schema": "topic-image-replay/readout-v2-fixed-choice/v1",
        "created_at": time.time(),
        "repo_root": str(repo_root),
        "git_base": "67764a929ebd0837e4cbf776739394faf6001ed2",
        "repo_snapshot": repo_git_snapshot(repo_root),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matrix_config": str(matrix_config),
        "matrix_config_sha256": hashlib.sha256(matrix_config.read_bytes()).hexdigest(),
        "model_key": args.model_key,
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


def expected_provenance(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        "records_sha256": str(manifest["records_sha256"]),
        "implementation_sha256": str(manifest["implementation_sha256"]),
        "matrix_config_sha256": str(manifest["matrix_config_sha256"]),
        "repo_head": str(manifest["repo_snapshot"]["head"]),
    }


def load_completed(path: Path, provenance: dict[str, str]) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("provenance") != provenance:
                raise RuntimeError(
                    f"Refusing to resume incompatible shard {path}: "
                    f"{row.get('provenance')} != {provenance}"
                )
            completed.add((str(row["dataset"]), str(row["sample_index"])))
    return completed


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    from vlmeval.dataset import build_dataset

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest["conditions"] != list(CONDITIONS):
        raise RuntimeError(f"Manifest condition mismatch: {manifest['conditions']}")
    matrix_config = Path(args.matrix_config or manifest["matrix_config"]).resolve()
    if hashlib.sha256(matrix_config.read_bytes()).hexdigest() != manifest["matrix_config_sha256"]:
        raise RuntimeError(f"Matrix config changed after manifest creation: {matrix_config}")
    provenance = expected_provenance(manifest)
    if args.smoke_one_per_dataset:
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
    completed = load_completed(output_jsonl, provenance) if args.resume else set()
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
                current_labels = row_choice_labels(dataset_name, row)
                current_answer = normalize_answer(row.get("answer"))
                if str(row.get("index")) != task["sample_index"]:
                    raise RuntimeError(f"Manifest/data index mismatch: {task} vs {row.get('index')}")
                if current_labels != task["choice_labels"] or current_answer != task["answer_key"]:
                    raise RuntimeError(f"Manifest/data label mismatch: {task}")
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


def read_jsonl_paths(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    provenance = expected_provenance(manifest)
    paths = sorted(Path(args.input_root).glob(args.glob))
    rows = read_jsonl_paths(paths)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_keys = []
    provenance_mismatches = []
    for row in rows:
        key = (str(row["dataset"]), str(row["sample_index"]))
        if row.get("provenance") != provenance:
            provenance_mismatches.append(key)
        if key in by_key:
            duplicate_keys.append(key)
        by_key[key] = row
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
        "duplicate_keys": [list(key) for key in duplicate_keys],
        "provenance_mismatches": [list(key) for key in provenance_mismatches],
        "complete": actual == expected and not duplicate_keys and not provenance_mismatches,
        "datasets": summaries,
    }
    output_root = Path(args.output_root)
    write_json(output_root / "summary.json", payload)
    (output_root / "accuracy.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    if args.require_complete and not payload["complete"]:
        raise SystemExit(2)
    return payload


def validate_smoke(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.raw_root)
    artifacts = sorted(root.glob("**/artifact.json"))
    rows = []
    for artifact_path in artifacts:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        allowed = np.load(artifact_path.parent / artifact["raw_files"]["allowed_masks"])
        seq_len = int(artifact["token_count"])
        decode_start = int(artifact["prefill_token_count"])
        i2 = artifact["readout_v2_span"]
        expected = allowed_masks(seq_len, decode_start, i2).numpy()
        plan = artifact["candidate_plan"]
        forced_prefix_ids = plan["forced_prefix_ids"]
        suffixes = {
            label: plan["joint_token_ids"][label][len(forced_prefix_ids) :]
            for label in artifact["choice_labels"]
        }
        seed_ids = artifact["boundary_meta"]["assistant_seed_token_ids"]
        token_table = artifact["token_table"]
        i2_tokens = token_table[int(i2["start"]) : int(i2["end"]) + 1]
        batch_logits = np.load(artifact_path.parent / artifact["raw_files"]["last_token_logits"])
        single_logits_path = artifact["raw_files"].get("last_token_logits_single4d")
        single_logits = np.load(artifact_path.parent / single_logits_path) if single_logits_path else None
        candidate_ids = [plan["candidate_token_ids"][label] for label in artifact["choice_labels"]]
        batch_single_diffs = (
            np.abs(batch_logits[:, candidate_ids] - single_logits[:, candidate_ids])
            if single_logits is not None
            else None
        )
        checks = {
            "raw_mask_exact": bool(np.array_equal(allowed, expected)),
            "content_is_iqi": artifact.get("content_counts") == {"image": 2, "text": 1},
            "forced_prefix_is_exactly_answer_colon": plan["forced_prefix_text"] == "Answer:",
            "candidate_suffixes_are_exactly_one_token": all(
                len(suffixes[label]) == 1
                and suffixes[label][0] == plan["candidate_token_ids"][label]
                for label in artifact["choice_labels"]
            ),
            "decode_restriction_starts_before_assistant_seed": (
                decode_start + len(seed_ids) + len(forced_prefix_ids) == seq_len
            ),
            "readout_span_is_visual_core_only": bool(i2_tokens)
            and all(item["token"] == "<|image_pad|>" for item in i2_tokens),
            "batch3_single4d_candidate_logits_close": batch_single_diffs is not None
            and bool(np.max(batch_single_diffs) <= BATCH_SINGLE_CANDIDATE_ATOL),
            "batch3_single4d_candidate_argmax_equal": single_logits is not None
            and all(
                int(np.argmax(batch_logits[index, candidate_ids]))
                == int(np.argmax(single_logits[index, candidate_ids]))
                for index in range(len(CONDITIONS))
            ),
            "full_4d_exact": bool(artifact["mask_checks"].get("full_exact")),
            "blocked_corruption_invariant": bool(
                artifact["diagnostics"].get("baseline_blocked_corruption_pass_1e_5")
            ),
        }
        checks["pass"] = all(checks.values())
        rows.append({"artifact": str(artifact_path), "dataset": artifact["dataset"], "checks": checks})
    datasets = {row["dataset"] for row in rows}
    payload = {
        "artifact_count": len(rows),
        "datasets": sorted(datasets),
        "expected_datasets": list(args.datasets),
        "all_datasets_present": datasets == set(args.datasets),
        "rows": rows,
    }
    payload["all_pass"] = payload["all_datasets_present"] and all(row["checks"]["pass"] for row in rows)
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
    parser = argparse.ArgumentParser(description="Fixed-choice IQI readout-v2 probe")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--repo-root", default=str(Path.cwd()))
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--datasets", default=",".join(DATASET_TARGETS))
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
    run.add_argument("--datasets", default=",".join(DATASET_TARGETS))
    run.add_argument("--model-key", default="qwen25vl_3b")
    run.add_argument("--model-path", default="")
    run.add_argument("--lmu-data", default=os.environ.get("LMUData", ""))
    run.add_argument("--matrix-config", default="")
    run.add_argument("--gpu-id", default="0")
    run.add_argument("--shard-rank", type=int, default=0)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--smoke-one-per-dataset", action="store_true")
    run.add_argument("--dump-raw-root", default="")
    run.add_argument("--diagnostics", action="store_true")

    summary = sub.add_parser("aggregate")
    summary.add_argument("--manifest", required=True)
    summary.add_argument("--input-root", required=True)
    summary.add_argument("--glob", default="shard*.jsonl")
    summary.add_argument("--output-root", required=True)
    summary.add_argument("--require-complete", action="store_true")

    validate = sub.add_parser("validate-smoke")
    validate.add_argument("--raw-root", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--datasets", default=",".join(DATASET_TARGETS))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "datasets"):
        args.datasets = parse_datasets(args.datasets)
    if args.command == "manifest":
        payload = make_manifest(args)
    elif args.command == "run":
        payload = run_shard(args)
    elif args.command == "aggregate":
        payload = aggregate(args)
    else:
        payload = validate_smoke(args)
    printable = payload
    if args.command == "manifest":
        printable = {
            "schema": payload["schema"],
            "record_count": payload["record_count"],
            "records_sha256": payload["records_sha256"],
            "datasets": payload["datasets"],
        }
    print(json.dumps(printable, ensure_ascii=False, default=json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
