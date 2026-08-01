from __future__ import annotations

import os
import sys
import warnings
import math
import logging
import json
import time
from typing import Any

import torch
from transformers import StoppingCriteria

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ..replay_policy import (
    apply_replay,
    canonicalize_replay_mode,
    is_noop_replay_mode,
    maybe_debug_print_replay,
    read_replay_config_from_env,
)
from ..final_model_input_dump import (
    dump_final_model_input,
    extract_replay_meta,
    final_input_dump_enabled,
    new_call_id,
    summarize_content_sequence,
    visual_spec,
)
from ..replay_image_transform import (
    apply_image_transform_to_content,
    canonicalize_image_transform,
)
from .replay_prompt_template import (
    apply_prompt_template_to_content,
    read_prompt_template_config_from_env,
    strip_prompt_template_from_content_for_direct_answer,
)
from ...smp import get_gpu_memory, listinstr
from ...dataset import DATASET_MODALITY

VLLM_MAX_IMAGE_INPUT_NUM = 24


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def create_image_content(image_path, min_pixels, max_pixels):
    base64_image, mime_type = encode_image(image_path)
    return {
        "type": "image",
        "image": f"data:{mime_type};base64,{base64_image}",
        'min_pixels': min_pixels,
        'max_pixels': max_pixels
    }


def encode_image(image_path, max_side=None):
    from mimetypes import guess_type
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"

    from PIL import Image
    image = Image.open(image_path)
    # Handle the alpha channel
    if image.mode == "RGBA":
        image = _rgba_to_rgb(image)
    if max_side:
        image = _resize_image(image, max_side)
    encoded_image = _encode_image(image, image_format)

    return encoded_image, mime_type


def _encode_image(image, image_format):
    from io import BytesIO
    with BytesIO() as output:
        image.convert("RGB").save(output, format=image_format)
        import base64
        base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
    return base64_encoded_data


def _rgba_to_rgb(image):
    from PIL import Image
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")


def _resize_image(image, max_side):
    resize_scale = max_side / max(image.size)
    new_size = (
        int(image.size[0] * resize_scale),
        int(image.size[1] * resize_scale),
    )
    return image.resize(new_size)


def process_video(video_path, num_frames, min_pixels, max_pixels):
    import cv2
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

    # the sampling rate using max number of frames
    sampling_gap_maxframe = (
        1 if not num_frames else math.ceil(frame_count / num_frames)
    )
    sampling_gap = max(math.ceil(fps / 5), sampling_gap_maxframe)

    frame_number = 0
    images = []

    while True:
        import tempfile
        success, frame = cap.read()
        if not success:
            break
        # Sample frames based on the dynamic sampling rate
        if frame_number % sampling_gap == 0:
            # Create a temporary file for the frame
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as temp_frame:
                cv2.imwrite(temp_frame.name, frame)
                images.append(create_image_content(temp_frame.name, min_pixels, max_pixels))
                os.remove(temp_frame.name)
        frame_number += 1
    if frame_number == 0:
        raise ValueError(f"Failed to read video from {video_path}, check data...")
    logging.info(
        f"Sampled {len(images)}/{frame_number} frames from video {video_path}"
    )
    cap.release()
    return images


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if (
                len(cur_keyword_ids) > 1
                and cur_keyword_ids[0] == tokenizer.bos_token_id
            ):
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [
            keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids
        ]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True
        )[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False


CHAT_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"  # noqa: E501

UNTIL = ["<|diff_marker|>"]


class Qwen2VLChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
        use_audio_in_video: bool = False,
        **kwargs,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        if self.total_pixels and self.total_pixels > 24576 * 28 * 28:
            print('The total number of video tokens might become too large, resulting in an overly long input sequence. We recommend lowering **total_pixels** to below **24576 × 28 × 28**.')  # noqa: E501
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        # assert False
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        if self.fps is None and self.nframe is None:
            print("Warning: fps and nframe are both None, \
                  using default nframe/fps setting in qwen-vl-utils/qwen-omni-utils, \
                  the fps/nframe setting in video dataset is omitted")
        self.use_audio_in_video = use_audio_in_video
        self.FRAME_FACTOR = 2
        assert model_path is not None
        self.model_path = model_path
        MODEL_CLS = None

        if listinstr(['omni'], model_path.lower()):
            try:
                from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
            except Exception as err:
                logging.critical("pip install git+https://github.com/huggingface/transformers@3a1ead0aabed473eafe527915eea8c197d424356")  # noqa: E501
                raise err
            MODEL_CLS = Qwen2_5OmniForConditionalGeneration
            self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        elif listinstr(['perceiver'], model_path.lower()):
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
        elif listinstr(['2.5', '2_5', 'qwen25', 'mimo'], model_path.lower()):
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
            MODEL_CLS = Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(model_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0
        self.use_vllm = kwargs.get('use_vllm', True)
        self.use_lmdeploy = kwargs.get('use_lmdeploy', False)
        env_limit_mm = os.environ.get("REPLAY_LIMIT_MM_PER_PROMPT", "").strip()
        if env_limit_mm.isdigit():
            self.limit_mm_per_prompt = max(1, int(env_limit_mm))
        else:
            self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM
        assert self.use_vllm + self.use_lmdeploy <= 1, "You can only set one flag between `use_vllm` and `use_lmdeploy` to True"  # noqa: E501

        if self.use_vllm:
            from vllm import LLM
            gpu_count = torch.cuda.device_count()
            env_max_model_len = os.environ.get("VLLM_MAX_MODEL_LEN", "").strip()
            env_max_num_seqs = os.environ.get("VLLM_MAX_NUM_SEQS", "").strip()
            # Allow explicit override by kwargs/env; otherwise auto-pick a valid TP size.
            tp_size = kwargs.get('tensor_parallel_size', None)
            if tp_size is None:
                env_tp_size = os.environ.get('VLLM_TP_SIZE', '').strip()
                tp_size = int(env_tp_size) if env_tp_size.isdigit() else None

            if tp_size is None:
                num_attention_heads = None
                try:
                    from transformers import AutoConfig
                    cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
                    num_attention_heads = getattr(cfg, 'num_attention_heads', None)
                    if num_attention_heads is None and hasattr(cfg, 'text_config'):
                        num_attention_heads = getattr(cfg.text_config, 'num_attention_heads', None)
                except Exception as err:
                    logging.warning(f'Failed to read num_attention_heads from config, fallback TP heuristic. {err}')

                if num_attention_heads is not None:
                    valid_tp = [tp for tp in range(min(gpu_count, int(num_attention_heads)), 0, -1)
                                if int(num_attention_heads) % tp == 0]
                    tp_size = valid_tp[0] if valid_tp else 1
                else:
                    if gpu_count >= 8:
                        tp_size = 8
                    elif gpu_count >= 4:
                        tp_size = 4
                    elif gpu_count >= 2:
                        tp_size = 2
                    else:
                        tp_size = 1

            tp_size = max(1, min(int(tp_size), max(1, gpu_count)))
            logging.info(
                f'Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})'
            )
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    'VLLM_WORKER_MULTIPROC_METHOD is not set to spawn.'
                    'Use \'export VLLM_WORKER_MULTIPROC_METHOD=spawn\' to avoid potential multi-process issues'
                )
            # Keep max_model_len configurable because some vLLM/Triton builds can fail
            # rotary/profile runs on very long context (e.g. 32768) for Qwen2-VL.
            max_model_len = kwargs.get("max_model_len", None)
            if max_model_len is None:
                max_model_len = int(env_max_model_len) if env_max_model_len.isdigit() else 8192
            max_num_seqs = int(env_max_num_seqs) if env_max_num_seqs.isdigit() else 5
            env_enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "").strip().lower()
            enforce_eager = env_enforce_eager in {"1", "true", "yes", "on"}
            gpu_utils = kwargs.get("gpu_utils", None)
            if gpu_utils is None:
                env_gpu_utils = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "").strip()
                gpu_utils = float(env_gpu_utils) if env_gpu_utils else 0.9
            self.llm = LLM(
                model=self.model_path,
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                limit_mm_per_prompt={"image": self.limit_mm_per_prompt},
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=gpu_utils,
                trust_remote_code=True,
                enforce_eager=enforce_eager,
            )

        elif self.use_lmdeploy:
            from lmdeploy import TurbomindEngineConfig, pipeline, ChatTemplateConfig
            num_gpus = torch.cuda.device_count()
            self.model = pipeline(
                model_path,
                backend_config=TurbomindEngineConfig(session_len=32768, cache_max_entry_count=0.1, tp=num_gpus),
                chat_template_config=ChatTemplateConfig(model_name='qwen2d5-vl'))
            torch.cuda.set_device(0)
            self.device = 'cuda'
        else:
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto', device_map="auto", trust_remote_code=True
            )
            self.model.eval()

        torch.cuda.empty_cache()

        legacy_stage_debug = os.environ.get("REPLAY_STAGE_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
        trace_level = os.environ.get("REPLAY_TRACE_LEVEL", "").strip().lower()
        if trace_level not in {"off", "summary", "full"}:
            if legacy_stage_debug or os.environ.get("REPLAY_DUMP_DIR", "").strip() or os.environ.get("REPLAY_PROMPT_AUDIT", "0").strip().lower() in {"1", "true", "yes", "on"}:
                trace_level = "summary"
            else:
                trace_level = "off"
        self._replay_trace_level = trace_level
        self._stage_debug_enabled = trace_level in {"summary", "full"}
        self._stage_debug_max_samples = int(
            os.environ.get(
                "REPLAY_TRACE_SAMPLES",
                os.environ.get("REPLAY_STAGE_DEBUG_SAMPLES", "3"),
            )
        )
        self._stage_debug_seen_samples = 0
        self._stage_debug_active = False

        self._replay_dump_dir = os.environ.get("REPLAY_TRACE_DIR", os.environ.get("REPLAY_DUMP_DIR", "")).strip()
        self._replay_dump_file = None
        self._replay_dump_max_chars = int(
            os.environ.get("REPLAY_TRACE_MAX_CHARS", os.environ.get("REPLAY_DUMP_MAX_CHARS", "0"))
        )
        self._prompt_audit_enabled = (
            os.environ.get("REPLAY_PROMPT_AUDIT", "0").strip().lower() in {"1", "true", "yes", "on"}
            or trace_level in {"summary", "full"}
        )
        self._prompt_audit_print = os.environ.get("REPLAY_PROMPT_AUDIT_PRINT", "0").strip().lower() in {"1", "true", "yes", "on"}
        self._last_trace_state: dict[str, Any] = {}
        if self._replay_dump_dir:
            os.makedirs(self._replay_dump_dir, exist_ok=True)
            self._replay_dump_file = os.path.join(self._replay_dump_dir, f"{self.__class__.__name__}.jsonl")
            print(f"[replay-dump] enabled. Writing to {self._replay_dump_file}", flush=True)

    def _trace_allows(self, detail: str = "summary") -> bool:
        if not self._stage_debug_active:
            return False
        if self._replay_trace_level == "off":
            return False
        if self._replay_trace_level == "full":
            return True
        return detail != "full"

    def _begin_stage_debug_sample(self):
        if self._stage_debug_enabled and self._stage_debug_seen_samples < self._stage_debug_max_samples:
            self._stage_debug_seen_samples += 1
            self._stage_debug_active = True
        else:
            self._stage_debug_active = False
        if not self._stage_debug_active:
            self._last_trace_state = {}

    def _stage_debug(self, stage: str, payload: dict, detail: str = "summary"):
        if not self._trace_allows(detail):
            return
        info = {"stage": stage, "model": self.__class__.__name__}
        info.update(payload)
        try:
            print("[REPLAY_TRACE] " + json.dumps(info, ensure_ascii=False), flush=True)
        except Exception:
            print(f"[REPLAY_TRACE] stage={stage} model={self.__class__.__name__}", flush=True)

    def _clip_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        max_chars = self._replay_dump_max_chars
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars] + f"\n...[TRUNCATED {len(text) - max_chars} chars]"
        return text

    @staticmethod
    def _has_boxed_instruction(text: str) -> bool:
        if not isinstance(text, str):
            return False
        lowered = text.lower()
        return ("\\boxed{" in text) or ("boxed{<answer>}" in lowered)

    @staticmethod
    def _has_one_line_instruction(text: str) -> bool:
        if not isinstance(text, str):
            return False
        return "return exactly one line in this format" in text.lower()

    def _collect_text_blocks(self, content: list[dict]) -> list[dict]:
        blocks = []
        for idx, item in enumerate(content):
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            text = item.get("text", "")
            blocks.append(
                {
                    "index": idx,
                    "chars": len(text),
                    "has_boxed_instruction": self._has_boxed_instruction(text),
                    "has_one_line_instruction": self._has_one_line_instruction(text),
                    "text": self._clip_text(text),
                }
            )
        return blocks

    def _summarize_prompt_flags(self, text: str) -> dict:
        return {
            "prompt_chars": len(text) if isinstance(text, str) else len(str(text)),
            "prompt_has_boxed_instruction": self._has_boxed_instruction(text),
            "prompt_has_one_line_instruction": self._has_one_line_instruction(text),
        }

    def _safe_image_ref(self, image_ref):
        if isinstance(image_ref, str):
            if image_ref.startswith("data:"):
                return f"{image_ref[:64]}...[len={len(image_ref)}]"
            return image_ref
        return str(image_ref)

    def _summarize_mm(self, mm_items):
        if mm_items is None:
            return []
        summary = []
        for idx, item in enumerate(mm_items):
            if isinstance(item, torch.Tensor):
                summary.append(
                    {
                        "index": idx,
                        "type": "tensor",
                        "shape": list(item.shape),
                        "dtype": str(item.dtype),
                        "device": str(item.device),
                    }
                )
            else:
                summary.append(
                    {
                        "index": idx,
                        "type": type(item).__name__,
                        "repr": self._clip_text(repr(item)),
                    }
                )
        return summary

    def _write_replay_dump(self, record: dict, detail: str = "summary"):
        if not self._replay_dump_file or not self._trace_allows(detail):
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
        except Exception as e:
            print(f"[replay-dump] write failed: {e}", flush=True)

    def _extract_replay_meta(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        return extract_replay_meta(inputs)

    def _dump_final_vllm_request(
        self,
        *,
        message,
        consumer_content,
        request,
        dataset,
        call_id,
        sampling_params=None,
        parent_call_id=None,
        batch_position=None,
    ):
        if not final_input_dump_enabled():
            return
        multi_modal_data = request.get("multi_modal_data", {}) if isinstance(request, dict) else {}
        source_refs = [
            str(item.get("image"))
            for item in consumer_content
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        visual_inputs = []
        if isinstance(multi_modal_data, dict):
            for modality, values in multi_modal_data.items():
                values = values if isinstance(values, (list, tuple)) else [values]
                for value in values:
                    position = len(visual_inputs)
                    source_ref = (
                        source_refs[position]
                        if str(modality) == "image" and position < len(source_refs)
                        else None
                    )
                    visual_inputs.append(
                        visual_spec(value, modality=str(modality), source_ref=source_ref)
                    )
        dump_final_model_input(
            model_family="qwen2.5-vl",
            backend="vllm",
            consumer_api="vllm.LLM.generate",
            text_chat_representation={
                "kind": "vllm_request_prompt",
                "value": request.get("prompt") if isinstance(request, dict) else request,
            },
            visual_inputs=visual_inputs,
            content_sequence=summarize_content_sequence(consumer_content),
            processor_inputs=request,
            generation_config=sampling_params,
            dataset=str(dataset) if dataset is not None else None,
            model_key=self.model_path,
            condition=getattr(self, "replay_cfg", {}).get("mode"),
            sample_meta=self._extract_replay_meta(message),
            call_id=call_id,
            parent_call_id=parent_call_id,
            batch_position=batch_position,
            observability={
                "boundary": "direct_vllm_generate_request",
                "post_dump_internal_processing": ["vLLM multimodal processor and model execution"],
            },
        )

    def _find_image_token_spans(self, input_ids: list[int]) -> list[dict[str, int]]:
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        spans = []
        start = None
        for idx, token_id in enumerate(input_ids):
            if token_id == image_token_id and start is None:
                start = idx
            elif token_id != image_token_id and start is not None:
                spans.append({"image_position": len(spans) + 1, "start": start, "end": idx - 1})
                start = None
        if start is not None:
            spans.append({"image_position": len(spans) + 1, "start": start, "end": len(input_ids) - 1})
        return spans

    def _summarize_token_ids(self, token_ids: list[int], detail: str = "summary") -> Any:
        if detail == "full" or len(token_ids) <= 256:
            return token_ids
        return {
            "length": len(token_ids),
            "head": token_ids[:128],
            "tail": token_ids[-128:],
        }

    def _record_processor_trace(
        self,
        *,
        text,
        images,
        videos,
        dataset: str | None,
        replayed_content: list[dict[str, Any]],
    ) -> None:
        if not self._trace_allows("summary"):
            return
        try:
            processor_inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
            )
        except Exception as err:
            self._write_replay_dump(
                {
                    "phase": "processor_trace_failed",
                    "dataset": str(dataset) if dataset is not None else None,
                    "error_type": type(err).__name__,
                    "error": self._clip_text(str(err)),
                },
                detail="summary",
            )
            return

        input_ids = processor_inputs.get("input_ids")
        attention_mask = processor_inputs.get("attention_mask")
        if input_ids is None or attention_mask is None:
            return
        prompt_ids = input_ids[0].tolist()
        prompt_mask = attention_mask[0].tolist()
        image_spans = self._find_image_token_spans(prompt_ids)
        target_span = image_spans[1] if len(image_spans) >= 2 else None
        self._last_trace_state = {
            "dataset": str(dataset) if dataset is not None else None,
            "prompt_token_count": len(prompt_ids),
            "prompt_token_ids": prompt_ids,
            "image_token_spans": image_spans,
            "target_image_span": target_span,
            "replayed_content": replayed_content,
        }
        token_payload = {
            "dataset": str(dataset) if dataset is not None else None,
            "prompt_token_count": len(prompt_ids),
            "attention_mask_count": int(sum(prompt_mask)),
            "prompt_token_ids": self._summarize_token_ids(prompt_ids, detail="full"),
            "attention_mask": self._summarize_token_ids(prompt_mask, detail="full"),
            "image_token_spans": image_spans,
            "target_image_span": target_span,
        }
        self._stage_debug("processor_inputs", token_payload, detail="full")
        self._write_replay_dump(
            {
                "phase": "processor_inputs",
                **token_payload,
                "processor_tensor_summary": {
                    key: {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                    for key, value in processor_inputs.items()
                    if hasattr(value, "shape") and hasattr(value, "dtype")
                },
            },
            detail="full",
        )

    def _record_generation_loss_mask(self, generated_text: str, dataset: str | None) -> None:
        if not self._trace_allows("full") or not self._last_trace_state:
            return
        prompt_ids = list(self._last_trace_state.get("prompt_token_ids", []))
        if not prompt_ids:
            return
        output_ids = self.processor.tokenizer(
            generated_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        prompt_len = len(prompt_ids)
        supervised_count = len(output_ids)
        labels = ([-100] * prompt_len) + list(output_ids)
        loss_mask = ([0] * prompt_len) + ([1] * supervised_count)
        payload = {
            "phase": "loss_mask",
            "dataset": str(dataset) if dataset is not None else None,
            "prompt_token_count": prompt_len,
            "generated_token_count": supervised_count,
            "loss_mask": self._summarize_token_ids(loss_mask, detail="full"),
            "labels": self._summarize_token_ids(labels, detail="full"),
            "supervised_token_span": None if supervised_count == 0 else {"start": prompt_len, "end": prompt_len + supervised_count - 1},
            "generated_token_ids": self._summarize_token_ids(list(output_ids), detail="full"),
        }
        self._stage_debug("loss_mask", payload, detail="full")
        self._write_replay_dump(payload, detail="full")

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
            elif s['type'] == 'video':
                item = {
                    'type': 'video',
                    'video': ensure_video_url(s['value'])
                }
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            elif s['type'] == 'audio':
                item = {'type':'audio','audio':s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def _prepare_content_vllm(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        video_inputs = [s for s in inputs if s['type'] == 'video']
        video_count = len(video_inputs)
        cur_image_count = 0
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                if cur_image_count < self.limit_mm_per_prompt:
                    content.append(item)
                    cur_image_count += 1
                else:
                    logging.warning(
                        f"Number of images exceeds the limit of {self.limit_mm_per_prompt}. "
                        f"Only the first {self.limit_mm_per_prompt} images will be used."
                    )
            elif s['type'] == 'video':
                if video_count > 1:
                    logging.warning(
                        "Multiple videos detected. Using video frames for each video"
                    )
                    if dataset == 'OCRBench':
                        min_pixels = 10 * 10 * 28 * 28
                        warnings.warn(f"OCRBench dataset uses custom min_pixels={min_pixels}")
                        if self.max_pixels is not None:
                            max_pixels = self.max_pixels
                    else:
                        if self.min_pixels is not None:
                            min_pixels = self.min_pixels
                        if self.max_pixels is not None:
                            max_pixels = self.max_pixels
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()

                    frames_per_video = max(1, self.limit_mm_per_prompt // video_count)
                    content.append({"type": "text", "text": "<video frames start>"})
                    content.extend(process_video(s['value'], frames_per_video, min_pixels, max_pixels))
                    content.append({"type": "text", "text": "<video frames end>"})

                else:
                    item = {
                        'type': 'video',
                        'video': ensure_video_url(s['value'])
                    }
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                    if self.total_pixels is not None:
                        item['total_pixels'] = self.total_pixels
                    if self.fps is not None:
                        item['fps'] = self.fps
                    elif self.nframe is not None:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            print(f"use {new_frame_count} for {s['value']}")
                            item['nframes'] = new_frame_count
                        else:
                            item['nframes'] = self.nframe
                    content.append(item)
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
                content.append(item)
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
        return content

    def generate_inner_transformers(self, message, dataset=None):
        if listinstr(['omni'], self.model_path.lower()):
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, please install it via 'pip install qwen-omni-utils[decord]'")  # noqa: E501
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        if listinstr(['omni'], self.model_path.lower()):
            audios, images, videos = process_mm_info([messages], use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(text=text, images=images,audio=audios, videos=videos, padding=True, return_tensors='pt',use_audio_in_video=self.use_audio_in_video)  # noqa: E501
        else:
            images, videos = process_vision_info([messages])
            inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')  # noqa: E501
        inputs = inputs.to('cuda')

        if final_input_dump_enabled():
            replay_meta = self._extract_replay_meta(message)
            image_refs = [
                item.get("image")
                for item in messages[-1].get("content", [])
                if isinstance(item, dict) and item.get("type") == "image"
            ]
            visual_inputs = [
                visual_spec(
                    image,
                    modality="image",
                    source_ref=image_refs[idx] if idx < len(image_refs) else None,
                )
                for idx, image in enumerate(images or [])
            ]
            visual_inputs.extend(visual_spec(video, modality="video") for video in (videos or []))
            dump_final_model_input(
                model_family="qwen2.5-vl",
                backend="transformers",
                consumer_api="transformers.PreTrainedModel.generate",
                text_chat_representation={"kind": "processor_chat_template", "value": text},
                visual_inputs=visual_inputs,
                content_sequence=summarize_content_sequence(message),
                processor_inputs=inputs,
                dataset=str(dataset) if dataset is not None else None,
                model_key=self.model_path,
                condition=getattr(self, "replay_cfg", {}).get("mode"),
                sample_meta=replay_meta,
                observability={
                    "boundary": "post_processor_device_inputs",
                    "post_dump_internal_processing": [],
                },
            )

        if listinstr(['omni'], self.model_path.lower()):
            self.generate_kwargs['use_audio_in_video'] = self.use_audio_in_video
            self.generate_kwargs['return_audio'] = False
        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response

    def generate_inner_lmdeploy(self, message, dataset=None):
        from lmdeploy import GenerationConfig
        gen_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            top_p=self.generate_kwargs['top_p'],
            top_k=self.generate_kwargs['top_k'],
            temperature=self.generate_kwargs['temperature'],
            repetition_penalty=self.generate_kwargs['repetition_penalty'],
        )
        gen_config.random_seed = None
        messages_list = self.message_to_lmdeploy(message, system_prompt=self.system_prompt)
        assert len(messages_list) == 1
        if final_input_dump_enabled():
            dump_final_model_input(
                model_family="qwen2.5-vl",
                backend="lmdeploy",
                consumer_api="lmdeploy.pipeline.__call__",
                text_chat_representation={"kind": "lmdeploy_messages", "value": messages_list},
                visual_inputs=[
                    visual_spec(item.get("value"), modality=item.get("type", "image"))
                    for item in message
                    if isinstance(item, dict) and item.get("type") in {"image", "video"}
                ],
                content_sequence=summarize_content_sequence(message),
                dataset=str(dataset) if dataset is not None else None,
                model_key=self.model_path,
                condition=getattr(self, "replay_cfg", {}).get("mode"),
                sample_meta=self._extract_replay_meta(message),
                observability={
                    "boundary": "lmdeploy_pipeline_call_arguments",
                    "post_dump_internal_processing": ["LMDeploy media loading and model preprocessing"],
                },
            )
        response = self.model(messages_list, gen_config=gen_config)[0]
        response = response.text
        return response

    def _build_vllm_sampling_params(self):
        from vllm import SamplingParams

        env = os.environ
        override_keys = {
            "temperature": "QWEN2VL_VLLM_TEMPERATURE",
            "top_p": "QWEN2VL_VLLM_TOP_P",
            "top_k": "QWEN2VL_VLLM_TOP_K",
            "repetition_penalty": "QWEN2VL_VLLM_REPETITION_PENALTY",
            "max_tokens": "QWEN2VL_VLLM_MAX_TOKENS",
        }
        if any(env.get(name, "").strip() for name in override_keys.values()):
            kwargs = {
                "temperature": float(env.get("QWEN2VL_VLLM_TEMPERATURE", "0.0")),
                "top_p": float(env.get("QWEN2VL_VLLM_TOP_P", "1.0")),
                "top_k": int(env.get("QWEN2VL_VLLM_TOP_K", "0")),
                "repetition_penalty": float(env.get("QWEN2VL_VLLM_REPETITION_PENALTY", "1.0")),
                "max_tokens": int(env.get("QWEN2VL_VLLM_MAX_TOKENS", str(self.max_new_tokens))),
            }
            stop_ids = env.get("QWEN2VL_VLLM_STOP_TOKEN_IDS", "").strip()
            if stop_ids:
                kwargs["stop_token_ids"] = [int(x) for x in stop_ids.replace(",", " ").split() if x]
            else:
                kwargs["stop_token_ids"] = None
            sampling_params = SamplingParams(**kwargs)
            print(f"using sampling_params: {sampling_params}", flush=True)
            return sampling_params

        return SamplingParams(
            temperature=0.0,
            max_tokens=self.max_new_tokens,
            stop_token_ids=None,
        )

    def _build_vllm_request(self, message, dataset=None, include_replayed_content=False):
        from vllm import SamplingParams

        if listinstr(['omni'], self.model_path.lower()):
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, please install it via 'pip install qwen-omni-utils[decord]'")  # noqa: E501
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
                raise err

        messages = []
        videos_nd = None
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content_vllm(message, dataset=dataset)})
        self._stage_debug(
            "after_prepare_content_vllm",
            {
                "dataset": str(dataset) if dataset is not None else None,
                "message_preview": str(messages)[:1200],
            },
        )
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if listinstr(['omni'], self.model_path.lower()):
            audios, images, videos = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
        else:
            images, videos = process_vision_info(messages)

        replayed_content = messages[-1].get("content", []) if messages else []
        replayed_image_refs = [
            self._safe_image_ref(item.get("image"))
            for item in replayed_content
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        image_pad_count = text.count("<|image_pad|>") if isinstance(text, str) else 0
        self._record_processor_trace(
            text=text,
            images=images,
            videos=videos,
            dataset=dataset,
            replayed_content=replayed_content,
        )
        self._write_replay_dump(
            {
                "phase": "prepared",
                "dataset": str(dataset) if dataset is not None else None,
                "replay_cfg": getattr(self, "replay_cfg", None),
                "prompt": self._clip_text(text),
                **self._summarize_prompt_flags(text),
                "prompt_image_pad_count": image_pad_count,
                "message_input": message,
                "message_replayed": replayed_content,
                "replayed_image_refs": replayed_image_refs,
                "replayed_image_count": len(replayed_image_refs),
                "vision_extract_image_count": len(images) if images is not None else 0,
                "vision_extract_video_count": len(videos) if videos is not None else 0,
                "vision_extract_image_summary": self._summarize_mm(images),
                "vision_extract_video_summary": self._summarize_mm(videos),
                "image_token_spans": self._last_trace_state.get("image_token_spans", []),
                "target_image_span": self._last_trace_state.get("target_image_span"),
            }
        )

        if replayed_image_refs and images is not None and len(images) < len(replayed_image_refs):
            logging.warning(
                "[replay-dump] Extracted image count (%d) is less than replayed image count (%d). "
                "This may indicate clipping by limit_mm_per_prompt or prompt formatting issue.",
                len(images),
                len(replayed_image_refs),
            )
        self._stage_debug(
            "after_chat_template_and_vision_info",
            {
                "dataset": str(dataset) if dataset is not None else None,
                **self._summarize_prompt_flags(text),
                "num_images": len(images) if images is not None else 0,
                "num_videos": len(videos) if videos is not None else 0,
                "prompt_preview": str(text)[:1200],
            },
        )

        if DATASET_MODALITY(dataset) == 'VIDEO' and 'megabench' not in dataset.lower():
            assert len(videos) == 1
            videos_nd = [videos[0].detach().cpu().numpy().transpose(0, 2, 3, 1)]

            video_inputs = {
                "prompt": text[0],
                "multi_modal_data": {"video": videos_nd[0]},
                "mm_processor_kwargs":{}
            }
            if self.use_audio_in_video:
                import vllm
                assert not vllm.envs.VLLM_USE_V1, ("V1 does not support use_audio_in_video. Please launch this example with `VLLM_USE_V1=0`.")  # noqa: E501
                video_inputs["multi_modal_data"]["audio"] = audios[0]
                video_inputs['mm_processor_kwargs']['use_audio_in_video'] = True
            if videos_nd[0].shape[0] > VLLM_MAX_IMAGE_INPUT_NUM:
                print('video input sequence may be too long for vllm, Maybe cannot generate response for VLLM')
        if images:
            req = {
                "prompt": text,
                "multi_modal_data": {"image": images},
            }
        elif videos_nd:
            req = video_inputs
        else:
            req = {
                "prompt": text,
            }
        if include_replayed_content:
            return req, replayed_content
        return req

    def _finalize_vllm_generated_text(self, generated_text, dataset=None):

        self._write_replay_dump(
            {
                "phase": "generated",
                "dataset": str(dataset) if dataset is not None else None,
                "replay_cfg": getattr(self, "replay_cfg", None),
                "output_text": self._clip_text(generated_text),
                "output_text_len": len(generated_text) if isinstance(generated_text, str) else None,
            }
        )
        self._record_generation_loss_mask(generated_text, dataset=dataset)
        if self.post_process:
            resp = generated_text.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                generated_text = resp[:end]

        if self.verbose:
            print(f'\033[32m{generated_text}\033[0m')
        return generated_text

    def generate_inner_vllm(self, message, dataset=None):
        sampling_params = self._build_vllm_sampling_params()
        if final_input_dump_enabled():
            req, consumer_content = self._build_vllm_request(
                message,
                dataset=dataset,
                include_replayed_content=True,
            )
            parent_call_id = new_call_id()
            self._dump_final_vllm_request(
                message=message,
                consumer_content=consumer_content,
                request=req,
                sampling_params=sampling_params,
                dataset=dataset,
                call_id=f"{parent_call_id}:0",
                parent_call_id=parent_call_id,
                batch_position=0,
            )
        else:
            req = self._build_vllm_request(message, dataset=dataset)
        outputs = self.llm.generate(req, sampling_params=sampling_params)
        generated_text = ''
        for o in outputs:
            generated_text = o.outputs[0].text
        return self._finalize_vllm_generated_text(generated_text, dataset=dataset)

    def generate_inner(self, message, dataset=None):
        self._begin_stage_debug_sample()
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        elif self.use_lmdeploy:
            return self.generate_inner_lmdeploy(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)

    def generate_batch_inner(self, messages, dataset=None):
        if self.use_vllm:
            self._begin_stage_debug_sample()
            if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], list):
                if DATASET_MODALITY(dataset) == 'VIDEO' and 'megabench' not in dataset.lower():
                    return [self.generate_inner_vllm(msg, dataset=dataset) for msg in messages]
                sampling_params = self._build_vllm_sampling_params()
                if final_input_dump_enabled():
                    built_requests = [
                        self._build_vllm_request(
                            msg,
                            dataset=dataset,
                            include_replayed_content=True,
                        )
                        for msg in messages
                    ]
                    reqs = [item[0] for item in built_requests]
                    consumer_contents = [item[1] for item in built_requests]
                else:
                    reqs = [self._build_vllm_request(msg, dataset=dataset) for msg in messages]
                    consumer_contents = [None] * len(messages)
                parent_call_id = new_call_id()
                for batch_position, (message, req, consumer_content) in enumerate(
                    zip(messages, reqs, consumer_contents)
                ):
                    self._dump_final_vllm_request(
                        message=message,
                        consumer_content=consumer_content,
                        request=req,
                        sampling_params=sampling_params,
                        dataset=dataset,
                        call_id=f"{parent_call_id}:{batch_position}",
                        parent_call_id=parent_call_id,
                        batch_position=batch_position,
                    )
                outputs = self.llm.generate(reqs, sampling_params=sampling_params)
                results = []
                for output in outputs:
                    generated_text = ''
                    if getattr(output, 'outputs', None):
                        generated_text = output.outputs[0].text
                    results.append(self._finalize_vllm_generated_text(generated_text, dataset=dataset))
                return results
            return [self.generate_inner_vllm(messages, dataset=dataset)]


class Qwen2VLChatReplay(Qwen2VLChat):
    """Replay-enabled Qwen2VLChat with minimal intrusion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_cfg = read_replay_config_from_env()
        self.prompt_template_cfg = read_prompt_template_config_from_env()
        self.template_on_last_replay_text = os.environ.get("REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        print(f"[Qwen2VLChatReplay] replay_cfg={self.replay_cfg}", flush=True)
        print(f"[Qwen2VLChatReplay] prompt_template_cfg={self.prompt_template_cfg}", flush=True)
        print(
            f"[Qwen2VLChatReplay] template_on_last_replay_text={self.template_on_last_replay_text}",
            flush=True,
        )
        self.safe_fallback_enabled = os.environ.get("REPLAY_SAFE_FALLBACK", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.safe_fallback_truncate_chars = max(
            512, int(os.environ.get("REPLAY_SAFE_TRUNCATE_CHARS", "6000"))
        )
        print(
            f"[Qwen2VLChatReplay] safe_fallback_enabled={self.safe_fallback_enabled} "
            f"safe_fallback_truncate_chars={self.safe_fallback_truncate_chars}",
            flush=True,
        )
        self.image_transform_name = canonicalize_image_transform(os.environ.get("REPLAY_IMAGE_TRANSFORM", "baseline"))
        self.image_transform_cache_dir = os.environ.get("REPLAY_IMAGE_TRANSFORM_CACHE_DIR", "").strip()
        self.image_transform_target_position = max(
            1,
            int(os.environ.get("REPLAY_IMAGE_TRANSFORM_TARGET_POSITION", "2")),
        )
        print(
            f"[Qwen2VLChatReplay] image_transform={self.image_transform_name} "
            f"target_position={self.image_transform_target_position} "
            f"cache_dir={self.image_transform_cache_dir or '<disabled>'}",
            flush=True,
        )

    def _apply_prompt_template_to_content(
        self,
        content: list[dict[str, str]],
        dataset: str | None = None,
    ) -> list[dict[str, str]]:
        templated = apply_prompt_template_to_content(
            content,
            self.prompt_template_cfg,
            dataset=dataset,
        )
        before_blocks = self._collect_text_blocks(content)
        after_blocks = self._collect_text_blocks(templated)
        audit_info = {
            "template_name": self.prompt_template_cfg.get("name"),
            "template_source": self.prompt_template_cfg.get("source"),
            "before_text_blocks": before_blocks,
            "after_text_blocks": after_blocks,
            "before_any_boxed_instruction": any(x.get("has_boxed_instruction") for x in before_blocks),
            "after_any_boxed_instruction": any(x.get("has_boxed_instruction") for x in after_blocks),
            "before_any_one_line_instruction": any(x.get("has_one_line_instruction") for x in before_blocks),
            "after_any_one_line_instruction": any(x.get("has_one_line_instruction") for x in after_blocks),
        }
        self._stage_debug(
            "prompt_template_before_after",
            {
                **audit_info,
                "before_preview": str(content)[:1200],
                "after_preview": str(templated)[:1200],
            },
        )
        if self._prompt_audit_enabled:
            self._write_replay_dump(
                {
                    "phase": "prompt_template_before_after",
                    **audit_info,
                }
            )
            if self._prompt_audit_print:
                print(
                    "[PROMPT_AUDIT] "
                    + json.dumps(
                        {
                            "template": self.prompt_template_cfg.get("name"),
                            "before_boxed": audit_info["before_any_boxed_instruction"],
                            "after_boxed": audit_info["after_any_boxed_instruction"],
                            "before_one_line": audit_info["before_any_one_line_instruction"],
                            "after_one_line": audit_info["after_any_one_line_instruction"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        return templated

    def _apply_replay_to_content(self, content: list[dict[str, str]]) -> list[dict[str, str]]:
        replayed = apply_replay(
            content,
            mode=self.replay_cfg["mode"],
            repeat_times=self.replay_cfg["repeat_times"],
            image_copy_mode=self.replay_cfg["image_copy_mode"],
        )
        self._stage_debug(
            "replay_before_after",
            {
                "replay_mode": self.replay_cfg["mode"],
                "before_preview": str(content)[:1200],
                "after_preview": str(replayed)[:1200],
            },
        )
        maybe_debug_print_replay(
            enabled=self.replay_cfg["debug"],
            mode=self.replay_cfg["mode"],
            before=content,
            after=replayed,
            tag=self.__class__.__name__,
        )
        return replayed

    def _apply_template_replay_pipeline(
        self,
        content: list[dict[str, str]],
        dataset: str | None = None,
    ) -> list[dict[str, str]]:
        # Default behavior: template first, then replay (legacy behavior).
        # Optional behavior: if replay is enabled and this flag is on,
        # replay first and template only the last text of replayed content.
        use_last_replay_text = getattr(self, "template_on_last_replay_text", False)
        replay_mode = canonicalize_replay_mode(self.replay_cfg.get("mode", "image_text"))
        if use_last_replay_text and not is_noop_replay_mode(replay_mode):
            replay_source = content
            if self.prompt_template_cfg.get("name") == "directly_answer":
                replay_source = strip_prompt_template_from_content_for_direct_answer(
                    content,
                    dataset=dataset,
                    text_key="text",
                )
            replayed = self._apply_replay_to_content(replay_source)
            return self._apply_prompt_template_to_content(replayed, dataset=dataset)

        templated = self._apply_prompt_template_to_content(content, dataset=dataset)
        return self._apply_replay_to_content(templated)

    def _apply_image_transform_pipeline(
        self,
        content: list[dict[str, str]],
        *,
        inputs: list[dict[str, Any]],
        dataset: str | None = None,
    ) -> list[dict[str, str]]:
        if self.image_transform_name == "baseline":
            return content
        replay_meta = self._extract_replay_meta(inputs)
        transformed, transform_record = apply_image_transform_to_content(
            content,
            transform_name=self.image_transform_name,
            sample_meta=replay_meta,
            cache_dir=self.image_transform_cache_dir or os.path.join(os.getcwd(), ".replay_transform_cache"),
            dataset_name=str(dataset) if dataset is not None else "unknown_dataset",
            image_position=self.image_transform_target_position,
        )
        self._stage_debug(
            "image_transform",
            {
                "dataset": str(dataset) if dataset is not None else None,
                "image_transform": self.image_transform_name,
                "record": transform_record,
            },
            detail="summary",
        )
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

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = super()._prepare_content(inputs, dataset=dataset)
        replayed = self._apply_template_replay_pipeline(content, dataset=dataset)
        return self._apply_image_transform_pipeline(replayed, inputs=inputs, dataset=dataset)

    def _prepare_content_vllm(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = super()._prepare_content_vllm(inputs, dataset=dataset)
        replayed = self._apply_template_replay_pipeline(content, dataset=dataset)
        return self._apply_image_transform_pipeline(replayed, inputs=inputs, dataset=dataset)

    def _run_with_replay_cfg(
        self,
        message: list[dict[str, str]],
        dataset: str | None,
        *,
        replay_mode: str,
        repeat_times: int,
        disable_image_transform: bool = False,
    ) -> str:
        old_cfg = dict(self.replay_cfg)
        old_transform_name = self.image_transform_name
        self.replay_cfg = dict(self.replay_cfg)
        self.replay_cfg["mode"] = replay_mode
        self.replay_cfg["repeat_times"] = repeat_times
        if disable_image_transform:
            self.image_transform_name = "baseline"
        try:
            return super().generate_inner_vllm(message, dataset=dataset)
        finally:
            self.replay_cfg = old_cfg
            self.image_transform_name = old_transform_name

    def _truncate_message_text(self, message: list[dict[str, str]]) -> list[dict[str, str]]:
        max_chars = self.safe_fallback_truncate_chars
        out = []
        for item in message:
            if isinstance(item, dict) and item.get("type") == "text":
                new_item = dict(item)
                txt = item.get("value", "")
                if isinstance(txt, str) and len(txt) > max_chars:
                    new_item["value"] = txt[:max_chars] + f"\n...[TRUNCATED {len(txt) - max_chars} chars]"
                out.append(new_item)
            else:
                out.append(item)
        return out

    def generate_inner_vllm(self, message, dataset=None):
        try:
            return super().generate_inner_vllm(message, dataset=dataset)
        except Exception as first_err:
            if not self.safe_fallback_enabled:
                raise

            replay_mode = canonicalize_replay_mode(self.replay_cfg.get("mode", "image_text"))
            print(
                f"[Qwen2VLChatReplay][safe-fallback] primary generation failed: {type(first_err).__name__}: {first_err}",
                flush=True,
            )

            if not is_noop_replay_mode(replay_mode):
                try:
                    out = self._run_with_replay_cfg(
                        message,
                        dataset,
                        replay_mode="image_text",
                        repeat_times=1,
                        disable_image_transform=True,
                    )
                    self._write_replay_dump(
                        {
                            "phase": "safe_fallback",
                            "strategy": "disable_replay",
                            "dataset": str(dataset) if dataset is not None else None,
                            "original_replay_mode": replay_mode,
                            "error_type": type(first_err).__name__,
                            "error": self._clip_text(str(first_err)),
                        }
                    )
                    return out
                except Exception as e_disable_replay:
                    print(
                        f"[Qwen2VLChatReplay][safe-fallback] disable_replay failed: {type(e_disable_replay).__name__}: {e_disable_replay}",
                        flush=True,
                    )

            try:
                trimmed_message = self._truncate_message_text(message)
                out = self._run_with_replay_cfg(
                    trimmed_message,
                    dataset,
                    replay_mode="image_text",
                    repeat_times=1,
                    disable_image_transform=True,
                )
                self._write_replay_dump(
                    {
                        "phase": "safe_fallback",
                        "strategy": "disable_replay_and_truncate_text",
                        "dataset": str(dataset) if dataset is not None else None,
                        "original_replay_mode": replay_mode,
                        "truncate_chars": self.safe_fallback_truncate_chars,
                        "error_type": type(first_err).__name__,
                        "error": self._clip_text(str(first_err)),
                    }
                )
                return out
            except Exception as second_err:
                print(
                    f"[Qwen2VLChatReplay][safe-fallback] truncate retry failed: {type(second_err).__name__}: {second_err}",
                    flush=True,
                )
                raise


class Qwen2VLChatAguvis(Qwen2VLChat):
    def __init__(self, mode=None, **kwargs):
        self.mode = mode
        super().__init__(**kwargs)
        self.processor.max_pixels = self.max_pixels
        self.processor.min_pixels = self.min_pixels

    def generate_inner(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical(
                "qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'"
            )
            raise err

        messages = []
        user_message = []
        for item in message:
            if "role" in item.keys():
                if item["role"] == "system":
                    self.system_prompt = item["value"]
                else:
                    item.pop("role")
                    user_message.append(item)
            else:
                user_message.append(item)
        message = user_message

        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {"role": "user", "content": self._prepare_content(message, dataset=dataset)}
        )
        if self.verbose:
            print(f"\033[31m{messages}\033[0m")

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            chat_template=CHAT_TEMPLATE,
        )
        # TODO: provide current action's low-level instruction
        # if False:
        #     # If low-level instruction is provided
        #     # We enforce using "Action: {low_level_instruction} to guide generation"
        #     recipient_text = f"<|im_start|>assistant<|recipient|>all\nAction: {low_level_instruction}\n"
        if self.mode == "force-plan":
            recipient_text = "<|im_start|>assistant<|recipient|>all\nThought: "
        elif self.mode == "force-plan-l1":
            recipient_text = "<|im_start|>assistant<|recipient|>all\nAction: "
        elif self.mode == "force-plan-l3":
            recipient_text = "<|im_start|>assistant<|recipient|>all\nObservation: "
        elif self.mode == "grounding":
            recipient_text = "<|im_start|>assistant<|recipient|>os\n"
        elif self.mode == "force-plan-free":
            recipient_text = "<|im_start|>assistant<|recipient|>all\n"
        elif self.mode == "self-plan":
            recipient_text = "<|im_start|>assistant<|recipient|>"
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        text += recipient_text
        # print(text)

        images, videos = process_vision_info([messages])
        inputs = self.processor(
            text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        # stop_str = "<|diff_marker|>"
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(
        #     keywords, self.processor.tokenizer, inputs.input_ids
        # )

        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
            # stopping_criteria=[stopping_criteria],
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        # for term in UNTIL:
        #     if len(term) > 0:
        #         response = response.split(term)[0]

        if self.post_process:
            resp = response.split("\\boxed{")[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == "{":
                    counter += 1
                elif resp[i] == "}":
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f"\033[32m{response}\033[0m")
        return response
