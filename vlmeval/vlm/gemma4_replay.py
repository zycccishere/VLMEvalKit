from __future__ import annotations

import logging
import os
import re
from typing import Any

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
    apply_replay,
    canonicalize_replay_mode,
    is_noop_replay_mode,
    maybe_debug_print_replay,
    read_replay_config_from_env,
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


def _env_optional_int(*names: str) -> int | None:
    for name in names:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            return int(raw)
        except Exception:
            continue
    return None


def _dataset_type(name: str | None) -> str | None:
    if not name:
        return None
    from ..dataset import DATASET_TYPE

    return DATASET_TYPE(name, default=None)


def _load_multimodal_model(model_path: str):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    try:
        from transformers import AutoModelForMultimodalLM
    except Exception:
        AutoModelForMultimodalLM = None

    loaders = []
    if AutoModelForMultimodalLM is not None:
        loaders.append(AutoModelForMultimodalLM)
    loaders.append(AutoModelForImageTextToText)
    loaders.append(AutoModelForCausalLM)

    base_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    last_err = None
    for loader in loaders:
        try:
            return loader.from_pretrained(
                model_path,
                attn_implementation="flash_attention_2",
                **base_kwargs,
            ).eval()
        except Exception as err:
            last_err = err
            logging.warning(
                "Gemma4 loader %s with flash_attention_2 failed for %s: %s: %s",
                getattr(loader, "__name__", str(loader)),
                model_path,
                type(err).__name__,
                err,
            )
            try:
                return loader.from_pretrained(model_path, **base_kwargs).eval()
            except Exception as err2:
                last_err = err2
                logging.warning(
                    "Gemma4 loader %s fallback failed for %s: %s: %s",
                    getattr(loader, "__name__", str(loader)),
                    model_path,
                    type(err2).__name__,
                    err2,
                )
    raise RuntimeError(f"Failed to load Gemma4 multimodal model from {model_path}: {last_err}")


class Gemma4Replay(Qwen3VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
        self,
        model_path: str,
        *,
        use_vllm: bool | None = None,
        tensor_parallel_size: int | None = None,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        limit_mm_per_prompt: int | None = None,
        gpu_utils: float = 0.85,
        max_new_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        repetition_penalty: float = 1.0,
        enable_thinking: bool | None = None,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.use_vllm = use_vllm if use_vllm is not None else _env_bool("GEMMA4_USE_VLLM", True)
        self.gpu_utils = float(kwargs.pop("gpu_utils", gpu_utils))
        self.max_new_tokens = int(
            kwargs.pop("max_new_tokens", _env_int("GEMMA4_MAX_NEW_TOKENS", max_new_tokens))
        )
        self.temperature = float(kwargs.pop("temperature", temperature))
        self.top_p = float(kwargs.pop("top_p", top_p))
        self.top_k = int(kwargs.pop("top_k", top_k))
        self.repetition_penalty = float(kwargs.pop("repetition_penalty", repetition_penalty))
        self.enable_thinking = (
            enable_thinking
            if enable_thinking is not None
            else _env_bool("GEMMA4_ENABLE_THINKING", False)
        )
        self.vllm_skip_special_tokens = _env_bool(
            "GEMMA4_VLLM_SKIP_SPECIAL_TOKENS",
            _env_bool("VLLM_SKIP_SPECIAL_TOKENS", not self.enable_thinking),
        )
        thinking_token_budget = kwargs.pop("thinking_token_budget", None)
        if thinking_token_budget is None:
            thinking_token_budget = _env_optional_int(
                "GEMMA4_VLLM_THINKING_TOKEN_BUDGET",
                "VLLM_THINKING_TOKEN_BUDGET",
            )
        self.vllm_thinking_token_budget = (
            int(thinking_token_budget) if thinking_token_budget and int(thinking_token_budget) > 0 else None
        )
        self.vllm_tensor_parallel_size = tensor_parallel_size
        self.vllm_max_model_len = max_model_len or _env_int(
            "GEMMA4_VLLM_MAX_MODEL_LEN",
            _env_int("VLLM_MAX_MODEL_LEN", 32768),
        )
        self.vllm_max_num_seqs = max_num_seqs or _env_int(
            "GEMMA4_VLLM_MAX_NUM_SEQS",
            _env_int("VLLM_MAX_NUM_SEQS", 4),
        )
        self.limit_mm_per_prompt = (
            limit_mm_per_prompt
            or _env_int(
                "REPLAY_LIMIT_MM_PER_PROMPT",
                _env_int("GEMMA4_VLLM_MAX_IMAGES", 24),
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
            llm_kwargs = {}
            if self.enable_thinking or self.vllm_thinking_token_budget is not None:
                llm_kwargs["reasoning_config"] = {
                    "reasoning_start_str": "<|channel>",
                    "reasoning_end_str": "<channel|>",
                }
            self.llm = LLM(
                model=self.model_path,
                trust_remote_code=True,
                tensor_parallel_size=tp_size,
                max_model_len=max(2048, int(self.vllm_max_model_len)),
                max_num_seqs=max(1, int(self.vllm_max_num_seqs)),
                limit_mm_per_prompt={"image": max(1, int(self.limit_mm_per_prompt))},
                gpu_memory_utilization=self.gpu_utils,
                seed=0,
                **llm_kwargs,
            )
            logging.info(
                "Using vLLM for Gemma4 %s with tp=%s max_model_len=%s max_num_seqs=%s skip_special_tokens=%s thinking_budget=%s",
                self.model_path,
                tp_size,
                self.vllm_max_model_len,
                self.vllm_max_num_seqs,
                self.vllm_skip_special_tokens,
                self.vllm_thinking_token_budget,
            )
        else:
            self.model = _load_multimodal_model(self.model_path)
            self.model.eval()

        self.replay_cfg = read_replay_config_from_env()
        self.prompt_template_cfg = read_prompt_template_config_from_env()
        self.template_on_last_replay_text = _env_bool("REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT", True)

    def _resolve_vllm_tp_size(self, explicit_tp_size: int | None = None) -> int:
        gpu_count = max(1, torch.cuda.device_count())
        tp_size = explicit_tp_size
        if tp_size is None:
            for env_name in ("GEMMA4_VLLM_TP_SIZE", "VLLM_TP_SIZE"):
                raw = os.environ.get(env_name, "").strip()
                if raw.isdigit():
                    tp_size = int(raw)
                    break
        if tp_size is None:
            tp_size = 1
        return max(1, min(int(tp_size), gpu_count))

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

    def _message_to_hf_messages(self, message):
        content = []
        for item in message:
            if item["type"] == "text":
                content.append({"type": "text", "text": str(item["value"])})
            elif item["type"] == "image":
                content.append({"type": "image", "url": str(item["value"])})
            else:
                raise ValueError(f"Gemma4Replay does not support content type: {item['type']}")
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        messages.append({"role": "user", "content": content})
        return messages

    def _message_to_vllm_payload(self, message):
        prompt_content = []
        images: list[Image.Image] = []
        for item in message:
            if item["type"] == "text":
                prompt_content.append({"type": "text", "text": str(item["value"])})
            elif item["type"] == "image":
                prompt_content.append({"type": "image", "image": ""})
                images.append(Image.open(item["value"]).convert("RGB"))
            else:
                raise ValueError(f"Gemma4Replay does not support content type: {item['type']}")
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        messages.append({"role": "user", "content": prompt_content})
        prompt = self._apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "multi_modal_data": {"image": images}}

    def _apply_chat_template(self, messages, **kwargs):
        try:
            return self.processor.apply_chat_template(
                messages,
                enable_thinking=self.enable_thinking,
                **kwargs,
            )
        except TypeError:
            return self.processor.apply_chat_template(messages, **kwargs)

    def _build_transformers_inputs(self, message):
        messages = self._message_to_hf_messages(message)
        inputs = self._apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_device = getattr(self.model, "device", "cuda")
        try:
            inputs = inputs.to(model_device)
            if hasattr(self.model, "dtype"):
                inputs = inputs.to(self.model.dtype)
        except Exception:
            inputs = inputs.to("cuda")
        return inputs

    def _build_sampling_params(self):
        from vllm import SamplingParams

        kwargs = dict(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_new_tokens,
            skip_special_tokens=self.vllm_skip_special_tokens,
        )
        if self.vllm_thinking_token_budget is not None:
            kwargs["thinking_token_budget"] = self.vllm_thinking_token_budget
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

    def _strip_reasoning_tags(self, text: str) -> str:
        cleaned = str(text)
        cleaned = re.sub(r"(?is)^<\|channel\>\s*thought\s*", "", cleaned, count=1)
        cleaned = re.sub(r"(?is)^<\|channel\>\s*response\s*", "", cleaned, count=1)
        cleaned = cleaned.replace("<channel|>", "")
        cleaned = re.sub(
            r"(?is)<\|?channel\|?>thought\s*<\|?channel\|?>.*?(?:<\|?channel\|?>response\s*<\|?channel\|?>)?",
            "",
            cleaned,
            count=1,
        )
        cleaned = re.sub(r"(?is)<think>\s*.*?\s*</think>\s*", "", cleaned, count=1)
        cleaned = cleaned.replace("<|channel|>response<|channel|>", "")
        cleaned = cleaned.replace("<|channel>response", "")
        cleaned = cleaned.replace("<|channel>thought", "")
        cleaned = cleaned.replace("<end_of_turn>", "")
        cleaned = cleaned.replace("<start_of_turn>model", "")
        return cleaned.strip()

    def _parse_vllm_reasoning_output(self, text: str) -> tuple[str, str] | None:
        raw = str(text)
        try:
            from vllm.reasoning.gemma4_utils import parse_thinking_output
        except Exception:
            return None
        try:
            parsed = parse_thinking_output(raw)
        except Exception as err:
            logging.warning("Gemma4 vLLM parse_thinking_output failed: %s: %s", type(err).__name__, err)
            return None
        if not isinstance(parsed, dict):
            return None
        thinking = str(parsed.get("thinking") or "").strip()
        answer = str(parsed.get("answer") or "").strip()
        if thinking:
            full_output = f"<think>\n{thinking}\n</think>"
            if answer:
                full_output = f"{full_output}\n{answer}"
            return answer or thinking, full_output
        if answer and answer != raw:
            return answer, raw
        return None

    def _parse_processor_response(self, text: str) -> tuple[str, str]:
        raw = str(text)
        vllm_parsed = self._parse_vllm_reasoning_output(raw)
        if vllm_parsed is not None:
            return vllm_parsed
        if hasattr(self.processor, "parse_response"):
            try:
                parsed = self.processor.parse_response(raw, return_response_as_dict=True)
            except TypeError:
                parsed = self.processor.parse_response(raw)
            except Exception as err:
                logging.warning("Gemma4 processor.parse_response failed: %s: %s", type(err).__name__, err)
                parsed = None

            if isinstance(parsed, dict):
                response = (
                    parsed.get("response")
                    or parsed.get("text")
                    or parsed.get("answer")
                    or parsed.get("content")
                    or ""
                )
                thought = (
                    parsed.get("thought")
                    or parsed.get("reasoning")
                    or parsed.get("internal_reasoning")
                    or ""
                )
                full_output = raw
                if thought and response:
                    full_output = f"<think>\n{thought}\n</think>\n{response}"
                return str(response or raw), str(full_output)
            if parsed not in (None, ""):
                return str(parsed), raw
        return raw, raw

    def _build_result(self, raw_text: str, dataset: str | None = None) -> dict[str, str]:
        parsed_response, full_output = self._parse_processor_response(raw_text)
        cleaned_response = self._strip_reasoning_tags(parsed_response)
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
            "full_output": str(full_output),
        }

    def _generate_inner_transformers(self, message, dataset=None):
        inputs = self._build_transformers_inputs(message)
        input_len = inputs["input_ids"].shape[-1]
        generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )
        generate_kwargs.update(self.extra_generate_kwargs)
        with torch.inference_mode():
            generation = self.model.generate(**inputs, **generate_kwargs)
        generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return self._build_result(decoded, dataset=dataset)

    def _generate_batch_inner_vllm(self, messages, dataset=None):
        sampling_params = self._build_sampling_params()
        requests = [self._message_to_vllm_payload(message) for message in messages]
        outputs = self.llm.generate(requests, sampling_params=sampling_params)
        return [
            self._build_result(output.outputs[0].text if getattr(output, "outputs", None) else "", dataset=dataset)
            for output in outputs
        ]

    def generate_inner(self, message, dataset=None):
        replayed = self._apply_replay_pipeline(message, dataset=dataset)
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
        replayed_messages = [
            self._apply_replay_pipeline(message, dataset=dataset)
            for message in messages
        ]
        if self.use_vllm:
            return self._generate_batch_inner_vllm(replayed_messages, dataset=dataset)
        return [self._generate_inner_transformers(message, dataset=dataset) for message in replayed_messages]
