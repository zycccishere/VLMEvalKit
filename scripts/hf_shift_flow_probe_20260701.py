#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from qwen25vl_image2_probe import (  # noqa: E402
    build_base_content,
    build_replayed_content,
    parse_attention_layers,
    sanitize_single_process_env,
    set_seed,
)
from vlmeval.dataset import build_dataset  # noqa: E402
from vlmeval.vlm.qwen2_vl.replay_prompt_template import (  # noqa: E402
    PROMPT_TEMPLATE_DIRECTLY_ANSWER,
    PROMPT_TEMPLATE_IDENTITY,
)
from vlmeval.vlm.replay_image_transform import apply_image_transform_to_content  # noqa: E402


TRANSFORMS_DEFAULT = [
    "shift_right_half_vit_token",
    "shift_right_one_vit_token",
    "shift_right_one_llm_token",
]


@dataclass(frozen=True)
class Span:
    name: str
    start: int
    end: int

    @property
    def token_count(self) -> int:
        return max(0, int(self.end) - int(self.start))

    def positions(self) -> list[int]:
        return list(range(int(self.start), int(self.end)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scalar I2->I1/Q1/I2 prefill-flow probe for HF Gemma3 and MiniCPM 4.5."
    )
    parser.add_argument("--model-family", required=True, choices=["gemma3", "minicpm-v-4_5", "minicpm-o-4_5"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-ids", nargs="*", default=[])
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--mode", default="image_text_image")
    parser.add_argument("--transforms", nargs="+", default=TRANSFORMS_DEFAULT)
    parser.add_argument(
        "--policy",
        default=PROMPT_TEMPLATE_IDENTITY,
        choices=[PROMPT_TEMPLATE_IDENTITY, PROMPT_TEMPLATE_DIRECTLY_ANSWER],
    )
    parser.add_argument("--template-on-last-replay-text", action="store_true")
    parser.add_argument("--attn-layers", default="last")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--scalar-raw-dump-limit", type=int, default=0)
    parser.add_argument("--scalar-query-chunk-size", type=int, default=256)
    parser.add_argument("--max-inp-length", type=int, default=8192)
    parser.add_argument(
        "--minicpm-max-slice-nums",
        type=int,
        default=1,
        help="MiniCPM smoke/probe default keeps each image to one 64-token visual slot.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Transform-pair manifest must be a JSON list.")
    return payload


def resolve_case_image_path(manifest_path: str | Path, image_ref: str) -> Path:
    image_path = Path(str(image_ref))
    if image_path.is_absolute() and image_path.exists():
        return image_path.resolve()
    if not image_path.is_absolute():
        candidate = (Path(manifest_path).resolve().parent / image_path).resolve()
        if candidate.exists():
            return candidate
        if image_path.exists():
            return image_path.resolve()
    raise FileNotFoundError(f"Case image not found: {image_ref}")


def resolve_dataset_row(dataset, dataset_name: str, sample_index: int) -> pd.Series:
    matches = dataset.data[dataset.data["index"] == sample_index]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row for {dataset_name} index={sample_index}, got {len(matches)}.")
    return matches.iloc[0]


def explicit_content_for_case(manifest_path: str | Path, case: dict[str, Any]) -> list[dict[str, Any]]:
    image_path = resolve_case_image_path(manifest_path, str(case["image"]))
    return [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": str(case["question"])},
    ]


def case_dataset_name(case: dict[str, Any]) -> str:
    if case.get("source_dataset"):
        return str(case["source_dataset"])
    source = case.get("source") if isinstance(case.get("source"), dict) else {}
    return str(source.get("dataset") or "controlled_target_box")


def case_sample_index(case: dict[str, Any], fallback: int) -> int | str:
    for key in ("source_index", "sample_index", "selection_rank"):
        if key in case:
            return case[key]
    source = case.get("source") if isinstance(case.get("source"), dict) else {}
    for key in ("row_index", "source_index"):
        if key in source:
            return source[key]
    return fallback


def unique_transforms(case: dict[str, Any], requested: list[str]) -> list[str]:
    transforms = ["baseline"]
    for item in requested:
        if item and item not in transforms:
            transforms.append(str(item))
    case_transform = str(case.get("shift_transform", "")).strip()
    if case_transform and case_transform not in transforms:
        transforms.append(case_transform)
    return transforms


def image_size_from_content(content: list[dict[str, Any]]) -> tuple[int, int] | None:
    for item in content:
        if item.get("type") != "image":
            continue
        ref = str(item.get("image") or item.get("value") or item.get("url") or "").strip()
        if ref.startswith("file://"):
            ref = ref[len("file://") :]
        if not ref:
            continue
        try:
            with Image.open(ref) as image:
                return tuple(int(v) for v in image.size)
        except Exception:
            return None
    return None


def open_rgb_image(ref: str) -> Image.Image:
    raw = str(ref or "").strip()
    if raw.startswith("file://"):
        raw = raw[len("file://") :]
    image = Image.open(raw)
    image.load()
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return image.convert("RGB")


def content_image_ref(item: dict[str, Any]) -> str:
    return str(item.get("image") or item.get("value") or item.get("url") or "").strip()


def strip_file_scheme(ref: str) -> str:
    raw = str(ref or "").strip()
    if raw.startswith("file://"):
        return raw[len("file://") :]
    return raw


def content_text_value(item: dict[str, Any]) -> str:
    return str(item.get("text") if "text" in item else item.get("value", ""))


def finite_mean(values: np.ndarray) -> float:
    values64 = np.asarray(values, dtype=np.float64)
    return float(values64.mean()) if values64.size else float("nan")


def safe_float16(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float16)


def resolve_input_device(model: Any, requested_device: str) -> torch.device:
    requested = str(requested_device).strip().lower()
    if requested != "auto":
        return torch.device(requested_device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_model_device(inputs: Any, device: torch.device) -> Any:
    if torch.is_tensor(inputs):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {key: to_model_device(value, device) for key, value in inputs.items()}
    if hasattr(inputs, "items"):
        return {key: to_model_device(value, device) for key, value in inputs.items()}
    if isinstance(inputs, list):
        return [to_model_device(value, device) for value in inputs]
    if isinstance(inputs, tuple):
        return tuple(to_model_device(value, device) for value in inputs)
    return inputs


def find_token_spans(input_ids: list[int], token_id: int) -> list[Span]:
    spans: list[Span] = []
    start: int | None = None
    for idx, current in enumerate(input_ids):
        if current == token_id and start is None:
            start = idx
        elif current != token_id and start is not None:
            spans.append(Span(name=f"image_{len(spans) + 1}", start=start, end=idx))
            start = None
    if start is not None:
        spans.append(Span(name=f"image_{len(spans) + 1}", start=start, end=len(input_ids)))
    return spans


def find_mid_text_positions(input_ids: list[int], image_spans: list[Span], special_token_ids: set[int]) -> list[int]:
    if len(image_spans) < 2:
        return []
    return [
        pos
        for pos in range(image_spans[0].end, image_spans[1].start)
        if input_ids[pos] not in special_token_ids
    ]


def tensor_shape_tree(value: Any, *, max_depth: int = 4) -> Any:
    if torch.is_tensor(value):
        return list(value.shape)
    if max_depth <= 0:
        return type(value).__name__
    if isinstance(value, (list, tuple)):
        return [tensor_shape_tree(item, max_depth=max_depth - 1) for item in list(value)[:4]]
    return type(value).__name__


def first_batch_item_count(value: Any) -> int | None:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        first = value[0]
        if isinstance(first, (list, tuple)):
            return len(first)
        if torch.is_tensor(first) and first.ndim >= 1:
            return int(first.shape[0])
    if torch.is_tensor(value) and value.ndim >= 1:
        return int(value.shape[0])
    return None


def token_id_if_available(tokenizer: Any, token: str) -> int | None:
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return None
    if token_id is None:
        return None
    try:
        token_id = int(token_id)
    except Exception:
        return None
    if token_id < 0:
        return None
    return token_id


def find_minicpm_q1_positions(
    input_ids: list[int],
    image_spans: list[Span],
    special_token_ids: set[int],
    tokenizer: Any,
) -> tuple[list[int], dict[str, Any]]:
    if len(image_spans) < 2:
        return [], {"strict_q1_positions": False}
    left = image_spans[0].end
    right = image_spans[1].start
    marker_ids = {
        token_id
        for token_id in [
            token_id_if_available(tokenizer, "<image_id>"),
            token_id_if_available(tokenizer, "<image>"),
            token_id_if_available(tokenizer, "<slice>"),
        ]
        if token_id is not None
    }
    cutoff = right
    for pos in range(left, right):
        if input_ids[pos] in marker_ids:
            cutoff = pos
            break
    positions = [pos for pos in range(left, cutoff) if input_ids[pos] not in special_token_ids]
    return positions, {
        "strict_q1_positions": True,
        "q1_range_start": int(left),
        "q1_range_stop": int(cutoff),
        "q1_raw_right_before_image2": int(right),
        "q1_tail_excluded_token_count": int(max(0, right - cutoff)),
        "q1_text_position_count": int(len(positions)),
        "q1_marker_token_ids": sorted(int(v) for v in marker_ids),
    }


def repeat_kv_local(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def apply_rotary_pos_emb_local(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (query_states * cos) + (rotate_half(query_states) * sin), (key_states * cos) + (
        rotate_half(key_states) * sin
    )


class ScalarFlowTracer:
    def __init__(
        self,
        attn_modules: dict[int, Any],
        *,
        family: str,
        scalar_query_chunk_size: int,
    ) -> None:
        self.attn_modules = attn_modules
        self.family = family
        self.scalar_query_chunk_size = int(scalar_query_chunk_size)
        self.original_forward: dict[int, Any] = {}
        self.records: list[dict[str, Any]] = []
        self.query_positions: list[int] = []
        self.image1_positions: list[int] = []
        self.text_positions: list[int] = []
        self.image2_positions: list[int] = []

    def configure_sample(
        self,
        *,
        query_positions: list[int],
        image1_positions: list[int],
        text_positions: list[int],
        image2_positions: list[int],
    ) -> None:
        self.query_positions = list(query_positions)
        self.image1_positions = list(image1_positions)
        self.text_positions = list(text_positions)
        self.image2_positions = list(image2_positions)

    def reset(self) -> None:
        self.records.clear()

    def patch(self) -> None:
        tracer = self
        for layer_idx, attn_module in self.attn_modules.items():
            self.original_forward[layer_idx] = attn_module.forward

            def instrumented_forward(
                module,
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor | None,
                past_key_values=None,
                cache_position: torch.LongTensor | None = None,
                _layer_idx: int = layer_idx,
                **kwargs,
            ):
                tracer.capture(
                    module=module,
                    layer_idx=_layer_idx,
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                )
                return tracer.original_forward[_layer_idx](
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **kwargs,
                )

            attn_module.forward = types.MethodType(instrumented_forward, attn_module)

    def restore(self) -> None:
        for layer_idx, attn_module in self.attn_modules.items():
            attn_module.forward = self.original_forward[layer_idx]

    def capture(
        self,
        *,
        module: Any,
        layer_idx: int,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any,
        cache_position: torch.LongTensor | None,
    ) -> None:
        if hidden_states.shape[0] != 1 or hidden_states.shape[1] <= 1 or not self.query_positions:
            return
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, int(module.head_dim))

        if self.family == "gemma3":
            query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            query_states = module.q_norm(query_states)
            key_states = module.k_norm(key_states)
        else:
            query_states = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_local(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states, _ = past_key_values.update(key_states, value_states, module.layer_idx, cache_kwargs)

        key_states_for_scores = repeat_kv_local(key_states, int(module.num_key_value_groups))

        query_chunks: list[list[int]]
        if self.scalar_query_chunk_size <= 0:
            query_chunks = [self.query_positions]
        else:
            chunk_size = max(1, int(self.scalar_query_chunk_size))
            query_chunks = [
                self.query_positions[start : start + chunk_size]
                for start in range(0, len(self.query_positions), chunk_size)
            ]

        chunk_payloads = [self._summarize_chunk(module, query_states, key_states_for_scores, attention_mask, chunk) for chunk in query_chunks if chunk]
        if not chunk_payloads:
            return

        self.records.append(
            {
                "layer": int(layer_idx),
                "query_count": int(len(self.query_positions)),
                "image1_key_count": int(len(self.image1_positions)),
                "text_key_count": int(len(self.text_positions)),
                "image2_key_count": int(len(self.image2_positions)),
                "scalar_query_chunk_size": int(self.scalar_query_chunk_size),
                "image1_mass_raw": np.concatenate([chunk["image1_mass_raw"] for chunk in chunk_payloads], axis=0),
                "text_mass_raw": np.concatenate([chunk["text_mass_raw"] for chunk in chunk_payloads], axis=0),
                "image2_mass_raw": np.concatenate([chunk["image2_mass_raw"] for chunk in chunk_payloads], axis=0),
            }
        )

    def _summarize_chunk(
        self,
        module: Any,
        query_states: torch.Tensor,
        key_states_for_scores: torch.Tensor,
        attention_mask: torch.Tensor | None,
        query_positions: list[int],
    ) -> dict[str, np.ndarray]:
        query_index = torch.as_tensor(query_positions, device=query_states.device, dtype=torch.long)
        query_slice = torch.index_select(query_states, dim=2, index=query_index)
        scores = torch.matmul(query_slice, key_states_for_scores.transpose(2, 3)) * module.scaling
        if attention_mask is not None:
            selected_mask = torch.index_select(
                attention_mask[:, :, :, : key_states_for_scores.shape[-2]],
                dim=2,
                index=query_index,
            )
            scores = scores + selected_mask
        if attention_mask is None and bool(getattr(module, "is_causal", False)):
            key_positions = torch.arange(key_states_for_scores.shape[-2], device=query_states.device, dtype=torch.long)
            causal_block = key_positions.view(1, 1, 1, -1) > query_index.view(1, 1, -1, 1)
            scores = scores.masked_fill(causal_block, torch.finfo(scores.dtype).min)
        attn_sel = F.softmax(scores, dim=-1, dtype=torch.float32)
        mean_attn = attn_sel.mean(dim=1).squeeze(0)

        image1_mass = self._sum_positions(mean_attn, self.image1_positions)
        text_mass = self._sum_positions(mean_attn, self.text_positions)
        image2_mass = self._sum_positions(mean_attn, self.image2_positions)
        return {
            "image1_mass_raw": image1_mass.detach().cpu().numpy(),
            "text_mass_raw": text_mass.detach().cpu().numpy(),
            "image2_mass_raw": image2_mass.detach().cpu().numpy(),
        }

    @staticmethod
    def _sum_positions(mean_attn: torch.Tensor, positions: list[int]) -> torch.Tensor:
        if not positions:
            return torch.zeros((mean_attn.shape[0],), device=mean_attn.device, dtype=mean_attn.dtype)
        index = torch.as_tensor(positions, device=mean_attn.device, dtype=torch.long)
        return torch.index_select(mean_attn, dim=1, index=index).sum(dim=-1)


class Gemma3Adapter:
    model_family = "gemma3"
    transform_family = "gemma3"

    def __init__(self, model_path: str, device: str) -> None:
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map=None,
        ).eval()
        self.device = resolve_input_device(self.model, device)
        self.model.to(self.device)
        self.tokenizer = self.processor.tokenizer
        self.special_token_ids = set(self.tokenizer.all_special_ids)
        self.image_token_id = int(getattr(self.tokenizer, "image_token_id", 262144))

    @property
    def layers(self) -> list[Any]:
        return list(self.model.model.language_model.layers)

    def content_to_inputs(self, content: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
        messages = [{"role": "user", "content": self._content_to_gemma_items(content)}]
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return {"prompt_text": prompt_text, "model_inputs": to_model_device(inputs, self.device)}

    def spans_from_inputs(self, inputs: dict[str, Any]) -> tuple[list[Span], list[int], dict[str, Any]]:
        input_ids = inputs["input_ids"][0].detach().cpu().tolist()
        spans = find_token_spans(input_ids, self.image_token_id)
        text_positions = find_mid_text_positions(input_ids, spans, self.special_token_ids)
        image_span_positions = set()
        for span in spans:
            image_span_positions.update(span.positions())
        token_type_ids = inputs.get("token_type_ids")
        token_type_image_positions: list[int] = []
        if torch.is_tensor(token_type_ids):
            token_type_image_positions = torch.where(token_type_ids[0].detach().cpu() == 1)[0].tolist()
        meta = {
            "span_source": "gemma_image_token_id",
            "image_token_id": self.image_token_id,
            "input_token_count": len(input_ids),
            "image_bound_count": len(spans),
            "pixel_values_shape": tensor_shape_tree(inputs.get("pixel_values")),
            "token_type_ids_shape": list(token_type_ids.shape) if torch.is_tensor(token_type_ids) else None,
            "token_type_image_count": len(token_type_image_positions),
            "token_type_image_matches_spans": set(token_type_image_positions) == image_span_positions,
            "q1_text_position_count": int(len(text_positions)),
        }
        return spans, text_positions, meta

    @staticmethod
    def _content_to_gemma_items(content: list[dict[str, Any]]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for item in content:
            if item.get("type") == "image":
                out.append({"type": "image", "url": strip_file_scheme(content_image_ref(item))})
            elif item.get("type") == "text":
                out.append({"type": "text", "text": content_text_value(item)})
            else:
                raise ValueError(f"Unsupported content item for Gemma3: {item}")
        return out


class MiniCPM45Adapter:
    transform_family = "minicpm45"

    def __init__(self, model_path: str, device: str, *, variant: str) -> None:
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        self.model_family = variant
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor.tokenizer = self.tokenizer
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        ).eval()
        self.device = resolve_input_device(self.model, device)
        self.model.to(self.device)
        self.special_token_ids = set(self.tokenizer.all_special_ids)

    @property
    def layers(self) -> list[Any]:
        return list(self.model.llm.model.layers)

    def content_to_inputs(self, content: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
        prompt_parts: list[str] = []
        images: list[Image.Image] = []
        image_marker = "(<image>./</image>)" if self.model_family == "minicpm-v-4_5" else "<image>./</image>"
        for item in content:
            if item.get("type") == "image":
                images.append(open_rgb_image(content_image_ref(item)))
                prompt_parts.append(image_marker)
            elif item.get("type") == "text":
                prompt_parts.append(content_text_value(item))
            else:
                raise ValueError(f"Unsupported content item for MiniCPM: {item}")
        messages = [{"role": "user", "content": "\n".join(prompt_parts)}]
        prompt_text = self._apply_chat_template(messages)
        if self.model_family == "minicpm-v-4_5":
            inputs = self.processor(
                [prompt_text],
                [images],
                max_slice_nums=args.minicpm_max_slice_nums,
                use_image_id=True,
                return_tensors="pt",
                max_length=args.max_inp_length,
            )
        else:
            inputs = self.processor(
                [prompt_text],
                [images],
                None,
                None,
                max_slice_nums=args.minicpm_max_slice_nums,
                use_image_id=True,
                stream_input=False,
                return_tensors="pt",
                max_length=args.max_inp_length,
            )
        inputs.pop("image_sizes", None)
        self._attach_position_ids(inputs)
        inputs = to_model_device(inputs, self.device)
        return {"prompt_text": prompt_text, "model_inputs": inputs}

    def spans_from_inputs(self, inputs: dict[str, Any]) -> tuple[list[Span], list[int], dict[str, Any]]:
        bounds = inputs["image_bound"][0]
        if torch.is_tensor(bounds):
            bounds_list = bounds.detach().cpu().tolist()
        else:
            bounds_list = [[int(pair[0]), int(pair[1])] for pair in bounds]
        spans = [
            Span(name=f"image_{idx + 1}", start=int(pair[0]), end=int(pair[1]))
            for idx, pair in enumerate(bounds_list)
        ]
        input_ids = inputs["input_ids"][0].detach().cpu().tolist()
        text_positions, q1_meta = find_minicpm_q1_positions(input_ids, spans, self.special_token_ids, self.tokenizer)
        meta = {
            "span_source": "minicpm_image_bound",
            "image_bound_count": len(spans),
            "image_bounds": [[span.start, span.end] for span in spans],
            "image_bound_lengths": [span.token_count for span in spans],
            "input_token_count": len(input_ids),
            "position_ids_shape": list(inputs["position_ids"].shape) if torch.is_tensor(inputs.get("position_ids")) else None,
            "position_ids_first": int(inputs["position_ids"][0, 0].detach().cpu().item())
            if torch.is_tensor(inputs.get("position_ids")) and inputs["position_ids"].numel()
            else None,
            "position_ids_last": int(inputs["position_ids"][0, -1].detach().cpu().item())
            if torch.is_tensor(inputs.get("position_ids")) and inputs["position_ids"].numel()
            else None,
            "minicpm_query_num": int(getattr(self.model.config, "query_num", 0) or 0),
            "minicpm_patch_size": int(getattr(self.model.config, "patch_size", 0) or 0),
            "minicpm_image_size": int(getattr(self.model.config, "image_size", 0) or 0),
            "pixel_values_shape": tensor_shape_tree(inputs.get("pixel_values")),
            "pixel_value_count": first_batch_item_count(inputs.get("pixel_values")),
            "tgt_sizes_shape": tensor_shape_tree(inputs.get("tgt_sizes")),
            "tgt_size_count": first_batch_item_count(inputs.get("tgt_sizes")),
            **q1_meta,
        }
        return spans, text_positions, meta

    def forward_prefill(self, inputs: dict[str, Any]) -> None:
        with torch.inference_mode():
            outputs = self.model(inputs, use_cache=False, return_dict=True)
        del outputs

    @staticmethod
    def _attach_position_ids(inputs: dict[str, Any]) -> None:
        if "input_ids" not in inputs:
            raise ValueError("MiniCPM inputs must contain input_ids.")
        inputs["input_ids"] = inputs["input_ids"].long()
        if "attention_mask" in inputs and torch.is_tensor(inputs["attention_mask"]):
            attention_mask = inputs["attention_mask"].long()
            position_ids = attention_mask.cumsum(dim=-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        else:
            seq_len = int(inputs["input_ids"].shape[-1])
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand_as(inputs["input_ids"])
        inputs["position_ids"] = position_ids.long()

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        if self.model_family == "minicpm-o-4_5":
            kwargs["use_tts_template"] = False
        try:
            return self.processor.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("use_tts_template", None)
            try:
                return self.processor.tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                kwargs.pop("enable_thinking", None)
                return self.processor.tokenizer.apply_chat_template(messages, **kwargs)


def load_adapter(model_family: str, model_path: str, device: str) -> Any:
    if model_family == "gemma3":
        return Gemma3Adapter(model_path, device)
    if model_family in {"minicpm-v-4_5", "minicpm-o-4_5"}:
        return MiniCPM45Adapter(model_path, device, variant=model_family)
    raise ValueError(f"Unsupported model family: {model_family}")


def gemma_forward_prefill(adapter: Gemma3Adapter, inputs: dict[str, Any]) -> None:
    with torch.inference_mode():
        outputs = adapter.model(**inputs, use_cache=False, return_dict=True)
    del outputs


def write_scalar_npz(
    output_dir: Path,
    rel_path: Path,
    *,
    record: dict[str, Any],
) -> None:
    np.savez_compressed(
        output_dir / rel_path,
        image1_mass_raw=safe_float16(record["image1_mass_raw"]),
        text_mass_raw=safe_float16(record["text_mass_raw"]),
        image2_mass_raw=safe_float16(record["image2_mass_raw"]),
    )


def build_layer_summary(
    *,
    record: dict[str, Any],
    scalar_npz_path: str,
) -> dict[str, Any]:
    image1_mass_raw = np.asarray(record["image1_mass_raw"], dtype=np.float32)
    text_mass_raw = np.asarray(record["text_mass_raw"], dtype=np.float32)
    image2_mass_raw = np.asarray(record["image2_mass_raw"], dtype=np.float32)
    mass_total = (
        np.asarray(image1_mass_raw, dtype=np.float64)
        + np.asarray(text_mass_raw, dtype=np.float64)
        + np.asarray(image2_mass_raw, dtype=np.float64)
    )
    return {
        "layer": int(record["layer"]),
        "dump_mode": "scalar",
        "npz_path": "",
        "scalar_npz_path": scalar_npz_path,
        "query_count": int(record.get("query_count", len(image2_mass_raw))),
        "image1_key_count": int(record.get("image1_key_count", 0)),
        "text_key_count": int(record.get("text_key_count", 0)),
        "image2_key_count": int(record.get("image2_key_count", 0)),
        "scalar_query_chunk_size": int(record.get("scalar_query_chunk_size", 0)),
        "mean_image1_mass_raw": finite_mean(image1_mass_raw),
        "mean_text_mass_raw": finite_mean(text_mass_raw),
        "mean_image2_mass_raw": finite_mean(image2_mass_raw),
        "mass_total_mean": finite_mean(mass_total),
        "mass_total_max": float(mass_total.max()) if mass_total.size else float("nan"),
        "position_band_mass": float("nan"),
        "content_band_mass": float("nan"),
        "expected_position_distance": float("nan"),
        "expected_content_distance": float("nan"),
        "expected_distance_from_diagonal": float("nan"),
        "row_entropy": float("nan"),
        "i2_total_self_mass_raw": finite_mean(image2_mass_raw),
        "i2_past_self_mass_raw": float("nan"),
        "i2_diag_self_mass_raw": float("nan"),
        "i2_local_self_mass_raw": float("nan"),
        "i2_local_self_ratio": float("nan"),
        "target_key_token_indices": [],
        "target_query_token_indices": [],
        "content_shifted_target_query_token_indices": [],
        "distractor_key_token_indices": [],
        "target_mass_norm_all_queries": float("nan"),
        "target_mass_norm_target_queries": float("nan"),
        "target_mass_norm_content_shifted_target_queries": float("nan"),
        "distractor_mass_norm_all_queries": float("nan"),
        "target_minus_distractor_mass": float("nan"),
        "distance_profile": [],
        "content_distance_profile": [],
    }


def main() -> int:
    args = build_parser().parse_args()
    sanitize_single_process_env()
    set_seed(args.seed)

    manifest = load_manifest(args.manifest)
    if args.case_ids:
        keep = set(args.case_ids)
        manifest = [item for item in manifest if item["id"] in keep]
    if args.max_cases > 0:
        manifest = manifest[: args.max_cases]
    if not manifest:
        raise ValueError("No cases selected from manifest.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scalar_npz_dir = output_dir / "scalar_npz"
    if args.scalar_raw_dump_limit > 0:
        scalar_npz_dir.mkdir(parents=True, exist_ok=True)

    adapter = load_adapter(args.model_family, args.model_path, args.device)
    selected_layers = parse_attention_layers(args.attn_layers, len(adapter.layers))
    tracer = ScalarFlowTracer(
        {layer_idx: adapter.layers[layer_idx].self_attn for layer_idx in selected_layers},
        family="gemma3" if args.model_family == "gemma3" else "qwen3",
        scalar_query_chunk_size=args.scalar_query_chunk_size,
    )
    tracer.patch()

    records_out: list[dict[str, Any]] = []
    dataset_cache: dict[str, Any] = {}
    run_start = time.perf_counter()

    try:
        for case_idx, case in enumerate(manifest):
            dataset_name = case_dataset_name(case)
            sample_index = case_sample_index(case, case_idx)
            if case.get("image") and case.get("question"):
                base_content = explicit_content_for_case(args.manifest, case)
            else:
                dataset = dataset_cache.get(dataset_name)
                if dataset is None:
                    dataset = build_dataset(dataset_name)
                    dataset_cache[dataset_name] = dataset
                row = resolve_dataset_row(dataset, dataset_name, int(sample_index))
                base_content = build_base_content(dataset, row)
            base_image_size = image_size_from_content(base_content)

            transforms_for_case = unique_transforms(case, args.transforms)
            transform_records: dict[str, Any] = {}
            layer_results: dict[str, list[dict[str, Any]]] = {}

            for transform in transforms_for_case:
                sample_start = time.perf_counter()
                replayed = build_replayed_content(
                    base_content,
                    dataset_name=dataset_name,
                    mode=args.mode,
                    policy=args.policy,
                    template_on_last_replay_text=args.template_on_last_replay_text,
                )
                transformed, transform_record = apply_image_transform_to_content(
                    replayed,
                    transform_name=transform,
                    sample_meta={"sample_index": sample_index},
                    cache_dir=output_dir / "_transform_cache" / transform / dataset_name,
                    dataset_name=dataset_name,
                    image_position=2,
                    model_family=adapter.transform_family,
                )
                prepared = adapter.content_to_inputs(transformed, args)
                model_inputs = prepared["model_inputs"]
                image_spans, text_positions, span_meta = adapter.spans_from_inputs(model_inputs)
                if len(image_spans) != 2:
                    raise ValueError(
                        f"Expected exactly 2 image spans for {case['id']} transform={transform}, got {len(image_spans)}."
                    )
                image1_positions = image_spans[0].positions()
                image2_positions = image_spans[1].positions()
                if not image1_positions or not image2_positions:
                    raise ValueError(f"Empty image span for {case['id']} transform={transform}.")

                tracer.configure_sample(
                    query_positions=image2_positions,
                    image1_positions=image1_positions,
                    text_positions=text_positions,
                    image2_positions=image2_positions,
                )
                tracer.reset()

                if args.model_family == "gemma3":
                    gemma_forward_prefill(adapter, model_inputs)
                else:
                    adapter.forward_prefill(model_inputs)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if not tracer.records:
                    raise RuntimeError(f"No flow records captured for {case['id']} transform={transform}.")

                transform_records[transform] = {
                    "prompt_text": prepared["prompt_text"],
                    "transform_record": transform_record,
                    "content_shift_meta": {
                        "dx_tokens": float((transform_record.get("shift") or {}).get("dx", 0.0))
                        / float((transform_record.get("shift") or {}).get("llm_visual_token_stride", 56.0)),
                        "dy_tokens": float((transform_record.get("shift") or {}).get("dy", 0.0))
                        / float((transform_record.get("shift") or {}).get("llm_visual_token_stride", 56.0)),
                        "llm_visual_token_stride": float(
                            (transform_record.get("shift") or {}).get("llm_visual_token_stride", 56.0)
                        ),
                    },
                    "seconds": float(time.perf_counter() - sample_start),
                    "span_meta": span_meta,
                    "image1_grid": {
                        "token_count": len(image1_positions),
                        "llm_grid_h": int(round(math.sqrt(len(image1_positions)))),
                        "llm_grid_w": int(round(math.sqrt(len(image1_positions)))),
                    },
                    "image2_grid": {
                        "token_count": len(image2_positions),
                        "llm_grid_h": int(round(math.sqrt(len(image2_positions)))),
                        "llm_grid_w": int(round(math.sqrt(len(image2_positions)))),
                    },
                }

                layer_summaries: list[dict[str, Any]] = []
                for record in tracer.records:
                    layer = int(record["layer"])
                    scalar_npz_rel = ""
                    if args.scalar_raw_dump_limit > 0 and case_idx < args.scalar_raw_dump_limit:
                        scalar_npz_rel_path = Path("scalar_npz") / f"{case['id']}__{transform}__layer{layer}.npz"
                        write_scalar_npz(output_dir, scalar_npz_rel_path, record=record)
                        scalar_npz_rel = str(scalar_npz_rel_path)
                    layer_summaries.append(build_layer_summary(record=record, scalar_npz_path=scalar_npz_rel))
                layer_results[transform] = sorted(layer_summaries, key=lambda item: item["layer"])
                tracer.reset()

            records_out.append(
                {
                    "case_id": case["id"],
                    "base_id": case.get("base_id", ""),
                    "question_id": case.get("question_id", ""),
                    "group": case.get("group", ""),
                    "source_dataset": dataset_name,
                    "source_index": sample_index,
                    "question": str(case.get("question", "")),
                    "answer": str(case.get("answer", "")),
                    "image": str(case.get("image", "")),
                    "image_size": list(base_image_size) if base_image_size else None,
                    "target_box_xyxy": case.get("target_box_xyxy"),
                    "distractor_box_xyxy": case.get("distractor_box_xyxy"),
                    "mode": args.mode,
                    "policy": args.policy,
                    "selected_layers": selected_layers,
                    "transforms": {
                        transform: {**transform_records[transform], "layers": layer_results[transform]}
                        for transform in transforms_for_case
                    },
                }
            )
            print(
                json.dumps(
                    {"event": "case_complete", "case_id": case["id"], "transforms": transforms_for_case},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        tracer.restore()

    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "model_path": args.model_path,
        "model_family": args.model_family,
        "mode": args.mode,
        "policy": args.policy,
        "attn_layers": args.attn_layers,
        "selected_layers": selected_layers,
        "transforms": args.transforms,
        "dump_mode": "scalar",
        "scalar_raw_dump_limit": int(args.scalar_raw_dump_limit),
        "scalar_query_chunk_size": int(args.scalar_query_chunk_size),
        "minicpm_max_slice_nums": int(args.minicpm_max_slice_nums),
        "case_count": len(records_out),
        "run_seconds": float(time.perf_counter() - run_start),
        "cases": records_out,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "run_complete", "case_count": len(records_out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
