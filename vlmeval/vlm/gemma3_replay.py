from __future__ import annotations

import logging
import json
import os
import re
import time
from typing import Any

import numpy as np
import torch
from PIL import Image

from .base import BaseModel
from .qwen2_vl.replay_prompt_template import (
    read_prompt_template_config_from_env,
    render_prompt_with_template,
    strip_prompt_template_from_content_for_direct_answer,
)
from .qwen3_vl.prompt import Qwen3VLPromptMixin
from .replay_policy import (
    REPLAY_IMAGE_TEXT_FILLER_TEXT,
    apply_replay,
    canonicalize_replay_mode,
    is_noop_replay_mode,
    maybe_debug_print_replay,
    read_replay_config_from_env,
)
from .replay_image_transform import (
    apply_image_transform_to_content,
    canonicalize_image_transform,
)
from .replay_text_filler import (
    SPACE_FILLER_LITERAL,
    TextFillerAudit,
    build_text_filler,
    count_filler_tokens,
    filler_literal_for,
    read_text_filler_type,
    tokenize_length,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw.isdigit():
        return int(raw)
    return default


def _env_float_optional(*names: str) -> float | None:
    for name in names:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            return float(raw)
        except ValueError:
            logging.warning("Ignoring invalid float env %s=%r", name, raw)
    return None


def _env_int_optional(*names: str) -> int | None:
    for name in names:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            return int(raw)
        except ValueError:
            logging.warning("Ignoring invalid int env %s=%r", name, raw)
    return None


def _dataset_type(name: str | None) -> str | None:
    if not name:
        return None
    from ..dataset import DATASET_TYPE

    return DATASET_TYPE(name, default=None)


def _load_gemma3_transformers_model(model_path: str):
    from transformers import Gemma3ForConditionalGeneration

    base_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    try:
        return Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            **base_kwargs,
        ).eval()
    except Exception as err:
        logging.warning(
            "Gemma3 flash_attention_2 load failed for %s: %s: %s; falling back to default attention",
            model_path,
            type(err).__name__,
            err,
        )
        return Gemma3ForConditionalGeneration.from_pretrained(model_path, **base_kwargs).eval()


class Gemma3Replay(Qwen3VLPromptMixin, BaseModel):
    """Gemma 3 image-text wrapper aligned with upstream VLMEvalKit plus replay support."""

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
        self,
        model_path: str = "google/gemma-3-4b-it",
        *,
        use_vllm: bool | None = None,
        tensor_parallel_size: int | None = None,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        limit_mm_per_prompt: int | None = None,
        gpu_utils: float = 0.9,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = "You are a helpful assistant. ",
        **kwargs: Any,
    ) -> None:
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.use_vllm = use_vllm if use_vllm is not None else _env_bool("GEMMA3_USE_VLLM", True)
        self.gpu_utils = float(kwargs.pop("gpu_utils", gpu_utils))
        self.max_new_tokens = int(
            kwargs.pop("max_new_tokens", _env_int("GEMMA3_MAX_NEW_TOKENS", max_new_tokens))
        )
        env_temperature = _env_float_optional("GEMMA3_VLLM_TEMPERATURE", "GEMMA3_TEMPERATURE")
        self.temperature = float(kwargs.pop("temperature", env_temperature if env_temperature is not None else temperature))
        self.sampling_top_p = _env_float_optional("GEMMA3_VLLM_TOP_P", "GEMMA3_TOP_P")
        self.sampling_top_k = _env_int_optional("GEMMA3_VLLM_TOP_K", "GEMMA3_TOP_K")
        self.sampling_repetition_penalty = _env_float_optional(
            "GEMMA3_VLLM_REPETITION_PENALTY",
            "GEMMA3_REPETITION_PENALTY",
        )
        self.vllm_tensor_parallel_size = tensor_parallel_size
        self.vllm_max_model_len = max_model_len or _env_int(
            "GEMMA3_VLLM_MAX_MODEL_LEN",
            _env_int("VLLM_MAX_MODEL_LEN", 32768),
        )
        self.vllm_max_num_seqs = max_num_seqs or _env_int(
            "GEMMA3_VLLM_MAX_NUM_SEQS",
            _env_int("VLLM_MAX_NUM_SEQS", 4),
        )
        self.limit_mm_per_prompt = (
            limit_mm_per_prompt
            or _env_int(
                "REPLAY_LIMIT_MM_PER_PROMPT",
                _env_int("GEMMA3_VLLM_MAX_IMAGES", 24),
            )
        )
        self.extra_generate_kwargs = dict(kwargs)

        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = None
        self.llm = None
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        if self.use_vllm:
            from vllm import LLM

            tp_size = self._resolve_vllm_tp_size(self.vllm_tensor_parallel_size)
            self.llm = LLM(
                model=self.model_path,
                trust_remote_code=True,
                tensor_parallel_size=tp_size,
                max_model_len=max(2048, int(self.vllm_max_model_len)),
                max_num_seqs=max(1, int(self.vllm_max_num_seqs)),
                limit_mm_per_prompt={"image": max(1, int(self.limit_mm_per_prompt))},
                gpu_memory_utilization=self.gpu_utils,
                seed=0,
            )
            logging.info(
                "Using vLLM for Gemma3 %s with tp=%s max_model_len=%s max_num_seqs=%s",
                self.model_path,
                tp_size,
                self.vllm_max_model_len,
                self.vllm_max_num_seqs,
            )
        else:
            self.model = _load_gemma3_transformers_model(self.model_path)
            self.model.eval()

        self.replay_cfg = read_replay_config_from_env()
        self.prompt_template_cfg = read_prompt_template_config_from_env()
        self.template_on_last_replay_text = _env_bool("REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT", True)
        self._text_filler_audit = TextFillerAudit(self.__class__.__name__)
        self._gemma3_model_config = None
        self._gemma3_image_expansion_delta_cache = None
        self.image_transform_name = canonicalize_image_transform(os.environ.get("REPLAY_IMAGE_TRANSFORM", "baseline"))
        self.image_transform_cache_dir = os.environ.get("REPLAY_IMAGE_TRANSFORM_CACHE_DIR", "").strip()
        self.image_transform_target_position = max(
            1,
            _env_int("REPLAY_IMAGE_TRANSFORM_TARGET_POSITION", 2),
        )
        self._last_image_transform_record: dict[str, Any] | None = None
        self._gemma_transform_records_by_message_id: dict[int, dict[str, Any]] = {}

        trace_level = os.environ.get("REPLAY_TRACE_LEVEL", os.environ.get("REPLAY_STAGE_DEBUG", "off")).strip().lower()
        if trace_level in {"1", "true", "yes", "on"}:
            trace_level = "summary"
        if trace_level not in {"off", "summary", "full"}:
            trace_level = "off"
        self._replay_trace_level = trace_level
        self._trace_max_samples = max(0, _env_int("REPLAY_TRACE_SAMPLES", 3))
        self._trace_seen_samples = 0
        self._trace_active = False
        self._replay_dump_dir = os.environ.get("REPLAY_TRACE_DIR", os.environ.get("REPLAY_DUMP_DIR", "")).strip()
        self._replay_dump_file = None
        self._replay_dump_max_chars = _env_int("REPLAY_TRACE_MAX_CHARS", _env_int("REPLAY_DUMP_MAX_CHARS", 0))
        if self._replay_dump_dir:
            os.makedirs(self._replay_dump_dir, exist_ok=True)
            self._replay_dump_file = os.path.join(self._replay_dump_dir, f"{self.__class__.__name__}.jsonl")
            print(f"[gemma3-replay-dump] enabled. Writing to {self._replay_dump_file}", flush=True)
        print(
            f"[Gemma3Replay] image_transform={self.image_transform_name} "
            f"target_position={self.image_transform_target_position} "
            f"cache_dir={self.image_transform_cache_dir or '<disabled>'}",
            flush=True,
        )

    def _resolve_vllm_tp_size(self, explicit_tp_size: int | None = None) -> int:
        gpu_count = max(1, torch.cuda.device_count())
        tp_size = explicit_tp_size
        if tp_size is None:
            for env_name in ("GEMMA3_VLLM_TP_SIZE", "VLLM_TP_SIZE"):
                raw = os.environ.get(env_name, "").strip()
                if raw.isdigit():
                    tp_size = int(raw)
                    break
        if tp_size is None:
            tp_size = 1
        return max(1, min(int(tp_size), gpu_count))

    def _begin_trace_sample(self) -> None:
        if self._replay_trace_level in {"summary", "full"} and self._trace_seen_samples < self._trace_max_samples:
            self._trace_seen_samples += 1
            self._trace_active = True
        else:
            self._trace_active = False

    def _trace_allows(self, detail: str = "summary") -> bool:
        if not self._trace_active or not self._replay_dump_file:
            return False
        if self._replay_trace_level == "full":
            return True
        return detail != "full"

    def _clip_text(self, text: Any) -> str:
        text = str(text)
        if self._replay_dump_max_chars > 0 and len(text) > self._replay_dump_max_chars:
            return text[: self._replay_dump_max_chars] + f"\n...[TRUNCATED {len(text) - self._replay_dump_max_chars} chars]"
        return text

    def _write_replay_dump(self, record: dict[str, Any], detail: str = "summary") -> None:
        if not self._trace_allows(detail):
            return
        payload = {
            "ts": time.time(),
            "model_class": self.__class__.__name__,
            "trace_level": self._replay_trace_level,
        }
        payload.update(record)
        try:
            with open(self._replay_dump_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as err:
            print(f"[gemma3-replay-dump] write failed: {err}", flush=True)

    @staticmethod
    def _safe_token(text: Any) -> str:
        out = []
        for ch in str(text):
            if ch.isalnum() or ch in {"-", "_", "."}:
                out.append(ch)
            else:
                out.append("_")
        return "".join(out).strip("_") or "sample"

    @staticmethod
    def _strip_file_scheme(image_ref: str) -> str:
        raw = str(image_ref or "").strip()
        if raw.startswith("file://"):
            return raw[len("file://") :]
        return raw

    @classmethod
    def _open_rgb_image(cls, image_ref: str) -> Image.Image:
        with Image.open(cls._strip_file_scheme(image_ref)) as image:
            return image.convert("RGB")

    @staticmethod
    def _shift_pil_wrap(image: Image.Image, *, dx: int, dy: int) -> Image.Image:
        source = np.asarray(image.convert("RGB"))
        shifted = np.roll(source, shift=dy, axis=0)
        shifted = np.roll(shifted, shift=dx, axis=1)
        return Image.fromarray(shifted.astype(np.uint8), mode="RGB")

    def _extract_replay_meta(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        for item in inputs:
            if isinstance(item, dict) and isinstance(item.get("replay_meta"), dict):
                return dict(item["replay_meta"])
        return {}

    def _processor_pixel_values(self, image: Image.Image):
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            return None
        outputs = image_processor(images=[image], return_tensors="pt")
        if hasattr(outputs, "get"):
            return outputs.get("pixel_values")
        return None

    def _record_gemma_image_transform_validation(
        self,
        *,
        images: list[Image.Image],
        dataset: str | None,
        transform_record: dict[str, Any] | None = None,
    ) -> None:
        transform_record = transform_record or self._last_image_transform_record
        if not isinstance(transform_record, dict) or not _env_bool("REPLAY_PROCESSOR_TRACE_VALIDATE", False):
            return
        shift = transform_record.get("shift")
        image_position = int(transform_record.get("target_image_position", 2) or 2)
        target_index = image_position - 1
        payload: dict[str, Any] = {
            "phase": "gemma_image_transform_validation",
            "dataset": str(dataset) if dataset is not None else None,
            "image_transform": transform_record.get("transform"),
            "sample_index": transform_record.get("sample_index"),
            "target_image_position": image_position,
            "transform_record": transform_record,
        }
        if not isinstance(shift, dict) or not shift.get("processed_space"):
            self._write_replay_dump({**payload, "ok": True, "comparison": "not_applicable"}, detail="summary")
            return
        if target_index < 0 or target_index >= len(images):
            self._write_replay_dump({**payload, "ok": False, "error": "target image outside vLLM payload"}, detail="summary")
            return
        processed_size = transform_record.get("processed_image_size_before_shift") or transform_record.get("transformed_image_size")
        if not isinstance(processed_size, list) or len(processed_size) != 2:
            self._write_replay_dump({**payload, "ok": False, "error": "missing processed image size"}, detail="summary")
            return
        try:
            original = self._open_rgb_image(str(transform_record.get("original_image_ref", "")))
            reference = original.resize((int(processed_size[0]), int(processed_size[1])), resample=Image.Resampling.BICUBIC)
            expected = self._shift_pil_wrap(
                reference,
                dx=int(shift.get("dx", 0) or 0),
                dy=int(shift.get("dy", 0) or 0),
            )
            actual = images[target_index].convert("RGB")
            actual_arr = np.asarray(actual)
            expected_arr = np.asarray(expected)
            same_image_shape = actual_arr.shape == expected_arr.shape
            if same_image_shape:
                image_diff = np.abs(actual_arr.astype(np.int16) - expected_arr.astype(np.int16))
                image_max_abs_diff = int(image_diff.max()) if image_diff.size else 0
                image_mean_abs_diff = float(image_diff.mean()) if image_diff.size else 0.0
            else:
                image_max_abs_diff = None
                image_mean_abs_diff = None
            processor_max_abs_diff = None
            processor_mean_abs_diff = None
            actual_pixels = self._processor_pixel_values(actual)
            expected_pixels = self._processor_pixel_values(expected)
            if actual_pixels is not None and expected_pixels is not None and tuple(actual_pixels.shape) == tuple(expected_pixels.shape):
                processor_diff = (actual_pixels.float() - expected_pixels.float()).abs()
                processor_max_abs_diff = float(processor_diff.max().item()) if processor_diff.numel() else 0.0
                processor_mean_abs_diff = float(processor_diff.mean().item()) if processor_diff.numel() else 0.0
            npz_path = ""
            if _env_bool("REPLAY_PROCESSOR_TRACE_SAVE_NPZ", False):
                out_dir = os.environ.get("REPLAY_PROCESSOR_TRACE_NPZ_DIR", self._replay_dump_dir or "").strip()
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                    npz_path = os.path.join(
                        out_dir,
                        "gemma_image_"
                        + "_".join(
                            [
                                self._safe_token(dataset or "dataset"),
                                self._safe_token(transform_record.get("sample_index", "sample")),
                                self._safe_token(transform_record.get("transform", "transform")),
                            ]
                        )
                        + ".npz",
                    )
                    np.savez_compressed(
                        npz_path,
                        actual_image=actual_arr,
                        expected_shifted_image=expected_arr,
                        reference_unshifted_image=np.asarray(reference),
                    )
            ok = bool(same_image_shape and image_max_abs_diff == 0)
            if processor_max_abs_diff is not None:
                ok = bool(ok and processor_max_abs_diff <= 1e-5)
            self._write_replay_dump(
                {
                    **payload,
                    "ok": ok,
                    "actual_image_size": list(actual.size),
                    "expected_image_size": list(expected.size),
                    "same_image_shape": same_image_shape,
                    "image_max_abs_diff_actual_vs_expected_shifted": image_max_abs_diff,
                    "image_mean_abs_diff_actual_vs_expected_shifted": image_mean_abs_diff,
                    "processor_max_abs_diff_actual_vs_expected_shifted": processor_max_abs_diff,
                    "processor_mean_abs_diff_actual_vs_expected_shifted": processor_mean_abs_diff,
                    "npz_path": npz_path,
                },
                detail="summary",
            )
        except Exception as err:
            self._write_replay_dump(
                {
                    **payload,
                    "ok": False,
                    "error_type": type(err).__name__,
                    "error": self._clip_text(str(err)),
                },
                detail="summary",
            )

    def _apply_image_transform_pipeline(
        self,
        message: list[dict[str, Any]],
        *,
        inputs: list[dict[str, Any]],
        dataset: str | None = None,
    ) -> list[dict[str, Any]]:
        if self.image_transform_name == "baseline":
            self._last_image_transform_record = None
            return message
        replay_meta = self._extract_replay_meta(inputs)
        transformed, transform_record = apply_image_transform_to_content(
            message,
            transform_name=self.image_transform_name,
            sample_meta=replay_meta,
            cache_dir=self.image_transform_cache_dir or os.path.join(os.getcwd(), ".replay_transform_cache"),
            dataset_name=str(dataset) if dataset is not None else "unknown_dataset",
            image_position=self.image_transform_target_position,
            model_family="gemma3",
        )
        self._last_image_transform_record = transform_record
        self._gemma_transform_records_by_message_id[id(transformed)] = transform_record
        self._write_replay_dump(
            {
                "phase": "image_transform",
                "dataset": str(dataset) if dataset is not None else None,
                "image_transform": self.image_transform_name,
                "replay_meta": replay_meta,
                "record": transform_record,
            },
            detail="summary",
        )
        return transformed

    def _apply_prompt_template_to_message(self, message, dataset=None):
        template_text = self.prompt_template_cfg.get("template", "{problem}")
        if template_text == "{problem}":
            return message

        target_idx = None
        for idx, item in enumerate(message):
            if isinstance(item, dict) and item.get("type") == "text":
                target_idx = idx

        if target_idx is None:
            return message

        out = []
        for idx, item in enumerate(message):
            if idx != target_idx:
                out.append(item)
                continue
            new_item = dict(item)
            new_item["value"] = render_prompt_with_template(
                str(item.get("value", "")),
                self.prompt_template_cfg,
                dataset=dataset,
            )
            out.append(new_item)
        return out

    def _apply_replay_pipeline(self, message, dataset=None):
        mode = canonicalize_replay_mode(self.replay_cfg.get("mode", "image_text"))
        if mode == REPLAY_IMAGE_TEXT_FILLER_TEXT:
            return self._apply_text_filler_replay(message, dataset=dataset)
        return self._apply_replay_pipeline_mode(message, dataset=dataset, mode=mode)

    def _apply_replay_pipeline_mode(self, message, dataset=None, mode=None):
        if mode is None:
            mode = canonicalize_replay_mode(self.replay_cfg.get("mode", "image_text"))

        if self.template_on_last_replay_text and not is_noop_replay_mode(mode):
            replay_source = message
            if self.prompt_template_cfg.get("name") == "directly_answer":
                replay_source = strip_prompt_template_from_content_for_direct_answer(
                    message,
                    dataset=dataset,
                    text_key="value",
                )
            replayed = apply_replay(
                replay_source,
                mode=mode,
                repeat_times=self.replay_cfg.get("repeat_times", 1),
                image_copy_mode=self.replay_cfg.get("image_copy_mode", "reuse_path"),
            )
            return self._apply_prompt_template_to_message(replayed, dataset=dataset)

        templated = self._apply_prompt_template_to_message(message, dataset=dataset)
        return apply_replay(
            templated,
            mode=mode,
            repeat_times=self.replay_cfg.get("repeat_times", 1),
            image_copy_mode=self.replay_cfg.get("image_copy_mode", "reuse_path"),
        )

    def _gemma3_model_config_lazy(self):
        if self._gemma3_model_config is None:
            from transformers import AutoConfig

            self._gemma3_model_config = AutoConfig.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        return self._gemma3_model_config

    def _gemma3_visual_core_tokens_per_image(self) -> int:
        per_image = int(getattr(self.processor, "image_seq_length", 0) or 0)
        if per_image <= 0:
            config = self._gemma3_model_config_lazy()
            per_image = int(getattr(config, "mm_tokens_per_image", 0) or 0)
        if per_image <= 0:
            raise RuntimeError(
                f"Gemma3 processor/config at {self.model_path} lack a positive per-image token count"
            )
        return per_image

    def _gemma3_image_expansion_delta(self) -> int:
        """Engine-side prompt growth per image placeholder.

        vLLM (model_executor/models/gemma3_mm.py Gemma3ProcessingInfo
        .get_image_repl) replaces the single ``<start_of_image>`` placeholder
        token with ``processor.full_image_sequence`` tokenized without
        special tokens. Pan-and-scan would add crop images and scaffold
        text; the shared checkpoint resolves ``do_pan_and_scan`` falsy, and
        any truthy value fails loudly here instead of silently mismatching.
        """
        if self._gemma3_image_expansion_delta_cache is None:
            image_processor = getattr(self.processor, "image_processor", None)
            if image_processor is not None and getattr(
                image_processor, "do_pan_and_scan", None
            ):
                raise RuntimeError(
                    "Gemma3 pan-and-scan is enabled in this checkpoint's image "
                    "processor config; visual core would include crops and the "
                    "filler audit does not model it. Refusing to continue."
                )
            tokenizer = self.processor.tokenizer
            replacement = tokenize_length(tokenizer, str(self.processor.full_image_sequence))
            placeholder = tokenize_length(tokenizer, str(self.processor.boi_token))
            self._gemma3_image_expansion_delta_cache = replacement - placeholder
            logging.info(
                "Gemma3 image placeholder expansion delta measured: %s tokens per image",
                self._gemma3_image_expansion_delta_cache,
            )
        return self._gemma3_image_expansion_delta_cache

    def _apply_text_filler_replay(self, message, dataset=None):
        """⟨V,T,F,T⟩: the ⟨V,T,T⟩ pipeline with a token-matched spaces block
        inserted between the two text halves.

        The Gemma chat template trims every text part and concatenates parts
        without separators, so a standalone spaces item would be trimmed to
        the empty string; the filler is therefore embedded inside a single
        merged text item (first text half + F + second text half), which
        renders identically to the ⟨V,T,T⟩ baseline plus F.
        """
        base = self._apply_replay_pipeline_mode(message, dataset=dataset, mode="image_text_text")
        text_positions = [
            idx
            for idx, item in enumerate(base)
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if not text_positions or len(text_positions) % 2 != 0:
            raise RuntimeError(
                f"Gemma3 filler mode expected an even number of text items after "
                f"image_text_text expansion, got {len(text_positions)}"
            )
        half = len(text_positions) // 2
        first_text = "".join(str(base[idx]["value"]) for idx in text_positions[:half])
        second_text = "".join(str(base[idx]["value"]) for idx in text_positions[half:])
        image_items = [
            item
            for item in base
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        if not image_items:
            raise RuntimeError("Gemma3 filler mode requires at least one image item")
        per_image_core = self._gemma3_visual_core_tokens_per_image()
        visual_core = per_image_core * len(image_items)
        filler_type = read_text_filler_type()
        filler_literal = filler_literal_for(filler_type)
        baseline_prompt = self._message_to_vllm_payload(base)["prompt"]
        baseline_local_tokens = tokenize_length(self.processor.tokenizer, baseline_prompt)
        baseline_ids = self.processor.tokenizer(baseline_prompt, add_special_tokens=False).input_ids
        baseline_pure_space = self._count_pure_space_tokens(baseline_ids)
        baseline_pure_literal = self._count_pure_literal_tokens(baseline_ids, filler_literal)
        text_positions_set = set(text_positions)
        first_text_position = text_positions[0]

        def build_merged_message(filler_value):
            merged_value = first_text + filler_value + second_text
            merged_item = None
            out = []
            for idx, item in enumerate(base):
                if idx == first_text_position:
                    merged_item = {"type": "text", "value": merged_value}
                    out.append(merged_item)
                elif idx in text_positions_set:
                    continue
                else:
                    out.append(item)
            return out

        def count_fn(filler_value):
            prompt = self._message_to_vllm_payload(build_merged_message(filler_value))["prompt"]
            return count_filler_tokens(self.processor.tokenizer, prompt, baseline_prompt)

        tag = f"Gemma3 dataset={dataset}"
        spec = build_text_filler(count_fn, visual_core, literal=filler_literal, tag=tag)
        filler_value = filler_literal * int(spec["chars"])
        filler_message = build_merged_message(filler_value)
        meta = {
            "dataset": str(dataset) if dataset is not None else None,
            "mode": REPLAY_IMAGE_TEXT_FILLER_TEXT,
            "filler_type": filler_type,
            "filler_literal": filler_literal,
            "image_count": len(image_items),
            "per_image_core": per_image_core,
            "visual_core_tokens": visual_core,
            "filler_chars": int(spec["chars"]),
            "filler_tokens_local": int(spec["tokens"]),
            "baseline_local_tokens": baseline_local_tokens,
            "baseline_pure_space_tokens": baseline_pure_space,
            "baseline_pure_literal_tokens": baseline_pure_literal,
            "chat_template": "gemma3-packaged",
        }
        for item in filler_message:
            if isinstance(item, dict) and item.get("type") == "text":
                item["text_filler_meta"] = dict(meta)
                break
        self._text_filler_audit.log_construction({**meta, "phase_detail": "constructed"})
        return filler_message

    def _count_pure_space_tokens(self, ids):
        """Engine-side count of tokens whose decoded text is entirely space
        characters (U+0020 only); see the MiniCPM wrapper for rationale."""
        return self._count_pure_literal_tokens(ids, " ")

    def _count_pure_literal_tokens(self, ids, literal):
        """Engine-side count of tokens whose decoded text consists solely of
        ``literal`` (generalizes the pure-space audit to the dots filler)."""
        count = 0
        for token_id in ids:
            text = self.processor.tokenizer.decode([token_id])
            if text and set(text) == {literal}:
                count += 1
        return count

    def _audit_text_filler_engine_outputs(self, messages, outputs, dataset=None):
        if not any(
            isinstance(item, dict) and item.get("text_filler_meta")
            for message in messages
            for item in message
        ):
            return
        config = self._gemma3_model_config_lazy()
        image_token_id = int(getattr(config, "image_token_id", -1))
        delta = self._gemma3_image_expansion_delta()
        for message, output in zip(messages, outputs):
            meta = next(
                (
                    item["text_filler_meta"]
                    for item in message
                    if isinstance(item, dict) and item.get("text_filler_meta")
                ),
                None,
            )
            if meta is None:
                continue
            ids = list(output.prompt_token_ids or [])
            engine_len = len(ids)
            image_token_count = sum(1 for token_id in ids if token_id == image_token_id)
            predicted_len = (
                int(meta["baseline_local_tokens"])
                + int(meta["image_count"]) * delta
                + int(meta["visual_core_tokens"])
            )
            filler_literal = str(meta.get("filler_literal", SPACE_FILLER_LITERAL))
            filler_engine_spaces = self._count_pure_space_tokens(ids) - int(
                meta.get("baseline_pure_space_tokens", 0)
            )
            filler_engine_literal = self._count_pure_literal_tokens(ids, filler_literal) - int(
                meta.get("baseline_pure_literal_tokens", meta.get("baseline_pure_space_tokens", 0))
            )
            record = {
                **{k: v for k, v in meta.items() if k != "phase_detail"},
                "engine_prompt_len": engine_len,
                "engine_prompt_len_predicted": predicted_len,
                "scaffold_offset": engine_len - predicted_len,
                "engine_image_token_count": image_token_count,
                "filler_tokens_engine_spaces": filler_engine_spaces,
                "filler_tokens_engine_literal": filler_engine_literal,
            }
            reasons = []
            if engine_len != predicted_len:
                reasons.append(
                    f"engine prompt len {engine_len} != predicted {predicted_len} "
                    f"(image expansion delta {delta})"
                )
            if image_token_count != int(meta["visual_core_tokens"]):
                reasons.append(
                    f"engine image token count {image_token_count} != visual core "
                    f"{meta['visual_core_tokens']}"
                )
            if not (-2 <= filler_engine_literal - int(meta["visual_core_tokens"]) <= 0):
                reasons.append(
                    f"engine pure-literal filler tokens {filler_engine_literal} deviate "
                    f"from {meta['visual_core_tokens']}"
                )
            record["engine_audit_pass"] = not reasons
            record["engine_audit_reason"] = "; ".join(reasons) or "ok"
            self._text_filler_audit.log_engine(record)

    def _message_to_hf_messages(self, message):
        content = []
        for item in message:
            if item["type"] == "text":
                content.append({"type": "text", "text": str(item["value"])})
            elif item["type"] == "image":
                content.append({"type": "image", "url": str(item["value"])})
            else:
                raise ValueError(f"Gemma3Replay does not support content type: {item['type']}")
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        messages.append({"role": "user", "content": content})
        return messages

    def _message_to_vllm_payload(self, message, dataset=None):
        prompt_content = []
        images: list[Image.Image] = []
        for item in message:
            if item["type"] == "text":
                prompt_content.append({"type": "text", "text": str(item["value"])})
            elif item["type"] == "image":
                if len(images) >= int(self.limit_mm_per_prompt):
                    continue
                prompt_content.append({"type": "image", "image": ""})
                images.append(self._open_rgb_image(item["value"]))
            else:
                raise ValueError(f"Gemma3Replay does not support content type: {item['type']}")
        if len([item for item in message if item.get("type") == "image"]) > int(self.limit_mm_per_prompt):
            logging.warning(
                "Gemma3Replay image count exceeds limit_mm_per_prompt=%s; extra images were ignored",
                self.limit_mm_per_prompt,
            )
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        messages.append({"role": "user", "content": prompt_content})
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        transform_record = self._gemma_transform_records_by_message_id.get(id(message), self._last_image_transform_record)
        self._write_replay_dump(
            {
                "phase": "gemma_vllm_payload",
                "dataset": str(dataset) if dataset is not None else None,
                "prompt": self._clip_text(prompt),
                "prompt_image_count": len(images),
                "payload_image_sizes": [list(image.size) for image in images],
                "message_replayed": message,
                "transform_record": transform_record if isinstance(transform_record, dict) else None,
            },
            detail="summary",
        )
        self._record_gemma_image_transform_validation(
            images=images,
            dataset=dataset,
            transform_record=transform_record if isinstance(transform_record, dict) else None,
        )
        return {"prompt": prompt, "multi_modal_data": {"image": images}}

    def _build_transformers_inputs(self, message):
        messages = self._message_to_hf_messages(message)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_device = getattr(self.model, "device", "cuda")
        try:
            inputs = inputs.to(model_device, dtype=torch.bfloat16)
        except Exception:
            inputs = inputs.to("cuda", dtype=torch.bfloat16)
        return inputs

    def _build_sampling_params(self):
        from vllm import SamplingParams

        kwargs = dict(
            temperature=max(0.0, float(self.temperature)),
            max_tokens=self.max_new_tokens,
        )
        if self.sampling_top_p is not None:
            kwargs["top_p"] = self.sampling_top_p
        if self.sampling_top_k is not None:
            kwargs["top_k"] = self.sampling_top_k
        if self.sampling_repetition_penalty is not None:
            kwargs["repetition_penalty"] = self.sampling_repetition_penalty
        # Optional per-request sampling determinism, shared knob with the
        # MiniCPM wrapper (VLMEVAL_SAMPLING_SEED). Unset (default) leaves
        # SamplingParams.seed at vLLM's default.
        sampling_seed = _env_int_optional("VLMEVAL_SAMPLING_SEED")
        if sampling_seed is not None:
            kwargs["seed"] = sampling_seed
        logging.info("Gemma3 vLLM SamplingParams: %s", kwargs)
        print(f"Gemma3 vLLM SamplingParams: {kwargs}", flush=True)
        return SamplingParams(**kwargs)

    def _extract_mcq_answer_from_free_form(self, text: str) -> str | None:
        text = str(text).strip()
        if re.fullmatch(r"[A-Ia-i]", text):
            return text.upper()
        patterns = [
            r"\\boxed\{\s*([A-Ia-i])\s*\}",
            r"Answer:\s*([A-Ia-i])(?![A-Za-z])",
            r"\boption\s*([A-Ia-i])(?![A-Za-z])",
            r"\banswer is\s*\(?([A-Ia-i])\)?(?![A-Za-z])",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                return matches[-1].upper()
        return None

    def _build_result(self, raw_text: str, dataset: str | None = None) -> dict[str, str]:
        cleaned_response = str(raw_text).strip()
        dataset_type = _dataset_type(dataset)
        prediction = cleaned_response
        if dataset_type == "MCQ":
            mcq = self._extract_mcq_answer_from_free_form(cleaned_response)
            if mcq:
                prediction = mcq
        elif dataset_type == "Y/N":
            lowered = cleaned_response.lower()
            if lowered.startswith("yes"):
                prediction = "Yes"
            elif lowered.startswith("no"):
                prediction = "No"
        return {
            "prediction": str(prediction).strip(),
            "description": "",
            "detailed_prediction": cleaned_response,
            "full_output": str(raw_text),
        }

    def _generate_inner_transformers(self, message, dataset=None):
        inputs = self._build_transformers_inputs(message)
        input_len = inputs["input_ids"].shape[-1]
        generate_kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=self.temperature > 0)
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
        generate_kwargs.update(self.extra_generate_kwargs)
        with torch.inference_mode():
            generation = self.model.generate(**inputs, **generate_kwargs)
        generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return self._build_result(decoded, dataset=dataset)

    def _generate_batch_inner_vllm(self, messages, dataset=None):
        sampling_params = self._build_sampling_params()
        requests = [self._message_to_vllm_payload(message, dataset=dataset) for message in messages]
        outputs = self.llm.generate(requests, sampling_params=sampling_params)
        self._audit_text_filler_engine_outputs(messages, outputs, dataset=dataset)
        return [
            self._build_result(output.outputs[0].text if getattr(output, "outputs", None) else "", dataset=dataset)
            for output in outputs
        ]

    def generate_inner(self, message, dataset=None):
        self._begin_trace_sample()
        replayed = self._apply_replay_pipeline(message, dataset=dataset)
        replayed = self._apply_image_transform_pipeline(replayed, inputs=message, dataset=dataset)
        maybe_debug_print_replay(
            enabled=self.replay_cfg.get("debug", False),
            mode=canonicalize_replay_mode(self.replay_cfg.get("mode", "image_text")),
            before=message,
            after=replayed,
            tag=self.__class__.__name__,
        )
        if self.use_vllm:
            return self._generate_batch_inner_vllm([replayed], dataset=dataset)[0]
        return self._generate_inner_transformers(replayed, dataset=dataset)

    def generate_batch_inner(self, messages, dataset=None):
        replayed_messages = []
        for message in messages:
            self._begin_trace_sample()
            replayed = self._apply_replay_pipeline(message, dataset=dataset)
            replayed_messages.append(self._apply_image_transform_pipeline(replayed, inputs=message, dataset=dataset))
        if self.use_vllm:
            return self._generate_batch_inner_vllm(replayed_messages, dataset=dataset)
        return [self._generate_inner_transformers(message, dataset=dataset) for message in replayed_messages]
