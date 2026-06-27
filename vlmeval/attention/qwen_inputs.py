from __future__ import annotations

import os
from typing import Any

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from vlmeval.probe_attention import Span, parse_attention_layers
from vlmeval.vlm.qwen2_vl.replay_prompt_template import (
    PROMPT_TEMPLATE_DIRECTLY_ANSWER,
    PROMPT_TEMPLATE_IDENTITY,
    apply_prompt_template_to_content,
    strip_prompt_template_from_content_for_direct_answer,
)
from vlmeval.vlm.replay_policy import apply_replay, canonicalize_replay_mode


SINGLE_PROCESS_DISTRIBUTED_ENV_KEYS = (
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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_single_process_env() -> None:
    for key in SINGLE_PROCESS_DISTRIBUTED_ENV_KEYS:
        os.environ.pop(key, None)


def load_model_and_processor(model_path: str, device: str):
    sanitize_single_process_env()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    device_map = "auto" if str(device).strip().lower() == "auto" else device
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return processor, model


def _prompt_template_cfg(name: str) -> dict[str, str]:
    if name == PROMPT_TEMPLATE_DIRECTLY_ANSWER:
        return {
            "name": PROMPT_TEMPLATE_DIRECTLY_ANSWER,
            "template": (
                "{problem}\n"
                "Answer directly with a single word or short phrase.\n"
                "Do not output any explanation, derivation, words, or extra symbols."
            ),
            "source": "attention_probe",
        }
    return {
        "name": PROMPT_TEMPLATE_IDENTITY,
        "template": "{problem}",
        "source": "attention_probe",
    }


def build_base_content(dataset: Any, row: pd.Series) -> list[dict[str, Any]]:
    base_message = dataset.build_prompt(row)
    content: list[dict[str, Any]] = []
    for item in base_message:
        item_type = item.get("type")
        if item_type == "image":
            content.append({"type": "image", "image": item.get("value")})
        elif item_type == "text":
            content.append({"type": "text", "text": item.get("value")})
        else:
            raise ValueError(f"Unsupported prompt item type: {item_type!r}")
    return content


def build_replayed_content(
    content: list[dict[str, Any]],
    dataset_name: str,
    *,
    mode: str,
    policy: str,
    template_on_last_replay_text: bool,
) -> list[dict[str, Any]]:
    cfg = _prompt_template_cfg(policy)
    replay_mode = canonicalize_replay_mode(mode)

    if template_on_last_replay_text and replay_mode != "image_text":
        replay_source = content
        if policy == PROMPT_TEMPLATE_DIRECTLY_ANSWER:
            replay_source = strip_prompt_template_from_content_for_direct_answer(
                content,
                dataset=dataset_name,
                text_key="text",
            )
        replayed = apply_replay(
            replay_source,
            mode=replay_mode,
            repeat_times=1,
            image_copy_mode="reuse_path",
        )
        return apply_prompt_template_to_content(replayed, cfg, dataset=dataset_name)

    templated = apply_prompt_template_to_content(content, cfg, dataset=dataset_name)
    return apply_replay(
        templated,
        mode=replay_mode,
        repeat_times=1,
        image_copy_mode="reuse_path",
    )


def build_inputs(
    processor: Any,
    content: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, dict[str, torch.Tensor]]:
    messages = [{"role": "user", "content": content}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    model_inputs = processor(
        text=prompt_text,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )
    return messages, prompt_text, model_inputs


def find_image_spans(input_ids: list[int], image_token_id: int) -> list[Span]:
    spans: list[Span] = []
    start = None
    for idx, token_id in enumerate(input_ids):
        if token_id == image_token_id and start is None:
            start = idx
        elif token_id != image_token_id and start is not None:
            spans.append(Span(name=f"image_{len(spans) + 1}", start=start, end=idx - 1))
            start = None
    if start is not None:
        spans.append(Span(name=f"image_{len(spans) + 1}", start=start, end=len(input_ids) - 1))
    return spans


def tensor_to_device(inputs: dict[str, torch.Tensor], device: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in inputs.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def resolve_input_device(model: Any, requested_device: str) -> str:
    if str(requested_device).strip().lower() != "auto":
        return requested_device
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cuda:0"
