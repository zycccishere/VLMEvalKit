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
from .final_model_input_dump import (
    dump_final_model_input,
    extract_replay_meta,
    final_input_dump_enabled,
    new_call_id,
    summarize_content_sequence,
    visual_spec,
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

    def _extract_replay_meta(self, message) -> dict[str, Any]:
        return extract_replay_meta(message)

    def _dump_final_vllm_request(
        self,
        *,
        message,
        request,
        dataset,
        call_id,
        sampling_params=None,
        parent_call_id=None,
        batch_position=None,
    ) -> None:
        if not final_input_dump_enabled():
            return
        multi_modal_data = request.get("multi_modal_data", {})
        image_values = multi_modal_data.get("image", []) if isinstance(multi_modal_data, dict) else []
        image_values = image_values if isinstance(image_values, (list, tuple)) else [image_values]
        source_refs = [
            str(item.get("value"))
            for item in message
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        visual_inputs = [
            visual_spec(
                image,
                modality="image",
                source_ref=source_refs[idx] if idx < len(source_refs) else None,
            )
            for idx, image in enumerate(image_values)
        ]
        dump_final_model_input(
            model_family="gemma3",
            backend="vllm",
            consumer_api="vllm.LLM.generate",
            text_chat_representation={
                "kind": "vllm_request_prompt",
                "value": request.get("prompt"),
            },
            visual_inputs=visual_inputs,
            content_sequence=summarize_content_sequence(message),
            processor_inputs=request,
            generation_config=sampling_params,
            dataset=str(dataset) if dataset is not None else None,
            model_key=self.model_path,
            condition=self.replay_cfg.get("mode"),
            sample_meta=self._extract_replay_meta(message),
            call_id=call_id,
            parent_call_id=parent_call_id,
            batch_position=batch_position,
            observability={
                "boundary": "direct_vllm_generate_request",
                "post_dump_internal_processing": ["vLLM multimodal processor and model execution"],
            },
        )

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
                raise ValueError(f"Gemma3Replay does not support content type: {item['type']}")
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
                if len(images) >= int(self.limit_mm_per_prompt):
                    continue
                prompt_content.append({"type": "image", "image": ""})
                images.append(Image.open(item["value"]).convert("RGB"))
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
        if final_input_dump_enabled():
            input_ids = inputs.get("input_ids")
            try:
                decoded_prompt = self.processor.tokenizer.batch_decode(
                    input_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except Exception:
                decoded_prompt = None
            image_refs = [
                str(item.get("value"))
                for item in message
                if isinstance(item, dict) and item.get("type") == "image"
            ]
            pixel_values = inputs.get("pixel_values")
            if (
                pixel_values is not None
                and hasattr(pixel_values, "shape")
                and len(pixel_values.shape) >= 4
                and int(pixel_values.shape[0]) == len(image_refs)
            ):
                visual_inputs = [
                    visual_spec(pixel_values[idx], modality="image", source_ref=image_refs[idx])
                    for idx in range(len(image_refs))
                ]
            else:
                visual_inputs = [visual_spec(ref, modality="image", source_ref=ref) for ref in image_refs]
            dump_final_model_input(
                model_family="gemma3",
                backend="transformers",
                consumer_api="transformers.PreTrainedModel.generate",
                text_chat_representation={
                    "kind": "decoded_processor_input_ids",
                    "value": decoded_prompt,
                },
                visual_inputs=visual_inputs,
                content_sequence=summarize_content_sequence(message),
                processor_inputs=inputs,
                dataset=str(dataset) if dataset is not None else None,
                model_key=self.model_path,
                condition=self.replay_cfg.get("mode"),
                sample_meta=self._extract_replay_meta(message),
                observability={
                    "boundary": "post_processor_device_inputs",
                    "post_dump_internal_processing": [],
                },
            )
        with torch.inference_mode():
            generation = self.model.generate(**inputs, **generate_kwargs)
        generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return self._build_result(decoded, dataset=dataset)

    def _generate_batch_inner_vllm(self, messages, dataset=None):
        sampling_params = self._build_sampling_params()
        requests = [self._message_to_vllm_payload(message) for message in messages]
        parent_call_id = new_call_id()
        for batch_position, (message, request) in enumerate(zip(messages, requests)):
            self._dump_final_vllm_request(
                message=message,
                request=request,
                sampling_params=sampling_params,
                dataset=dataset,
                call_id=f"{parent_call_id}:{batch_position}",
                parent_call_id=parent_call_id,
                batch_position=batch_position,
            )
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
        replayed_messages = []
        replay_mode = canonicalize_replay_mode(self.replay_cfg.get("mode", "image_text"))
        for batch_position, message in enumerate(messages):
            replayed = self._apply_replay_pipeline(message, dataset=dataset)
            maybe_debug_print_replay(
                enabled=self.replay_cfg.get("debug", False),
                mode=replay_mode,
                before=message,
                after=replayed,
                tag=f"{self.__class__.__name__}[batch={batch_position};dataset={dataset}]",
            )
            replayed_messages.append(replayed)
        if self.use_vllm:
            return self._generate_batch_inner_vllm(replayed_messages, dataset=dataset)
        return [self._generate_inner_transformers(message, dataset=dataset) for message in replayed_messages]
