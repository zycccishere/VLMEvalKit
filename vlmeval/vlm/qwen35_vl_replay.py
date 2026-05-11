from __future__ import annotations

import json
import logging
import os
import time
import warnings

import torch

from .base import BaseModel
from .qwen3_vl.prompt import Qwen3VLPromptMixin
from .replay_policy import (
    apply_replay,
    canonicalize_replay_mode,
    is_noop_replay_mode,
    maybe_debug_print_replay,
    read_replay_config_from_env,
)
from .qwen2_vl.replay_prompt_template import (
    apply_prompt_template_to_content,
    read_prompt_template_config_from_env,
    strip_prompt_template_from_content_for_direct_answer,
)
from ..smp import get_gpu_memory, listinstr

VLLM_MAX_IMAGE_INPUT_NUM = 24
QWEN35_GENERAL_DATASETS = {"AI2D_TEST", "OCRBench", "SEEDBench2_Plus"}


def qwen35_is_general_dataset(dataset: str | None) -> bool:
    return (dataset or "").strip() in QWEN35_GENERAL_DATASETS


def resolve_qwen35_generation_profile(
    dataset: str | None,
    *,
    force_no_thinking: bool = False,
    max_new_tokens: int = 32768,
    thinking_temperature: float = 1.0,
    thinking_top_p: float = 0.95,
    thinking_top_k: int = 20,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 1.5,
) -> dict[str, int | float | bool]:
    if force_no_thinking:
        return {
            'enable_thinking': False,
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 20,
            'min_p': 0.0,
            'repetition_penalty': 1.0,
            'presence_penalty': 1.5,
            'max_new_tokens': max_new_tokens,
        }
    if qwen35_is_general_dataset(dataset):
        return {
            'enable_thinking': False,
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 20,
            'min_p': 0.0,
            'repetition_penalty': 1.0,
            'presence_penalty': 1.5,
            'max_new_tokens': max_new_tokens,
        }
    return {
        'enable_thinking': False,
        'temperature': thinking_temperature,
        'top_p': thinking_top_p,
        'top_k': thinking_top_k,
        'min_p': 0.0,
        'repetition_penalty': repetition_penalty,
        'presence_penalty': presence_penalty,
        'max_new_tokens': max_new_tokens,
    }


def is_moe_model(model_path: str) -> bool:
    import re
    return re.search(r'-A\d+B', model_path) is not None


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def read_num_attention_heads_from_raw_config(model_path: str) -> int | None:
    config_path = os.path.join(model_path, 'config.json')
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception as err:
        logging.warning(f'Failed to read raw config for TP heuristic from {config_path}: {type(err).__name__}: {err}')
        return None

    candidates = [cfg.get('num_attention_heads')]
    text_cfg = cfg.get('text_config')
    if isinstance(text_cfg, dict):
        candidates.append(text_cfg.get('num_attention_heads'))

    for value in candidates:
        if isinstance(value, int) and value > 0:
            return value
    return None


def resolve_vllm_tp_size(model_path: str, gpu_count: int, explicit_tp_size: int | None = None) -> int:
    tp_size = explicit_tp_size
    if tp_size is None:
        for env_name in ('QWEN35_VLLM_TP_SIZE', 'VLLM_TP_SIZE'):
            env_tp_size = os.environ.get(env_name, '').strip()
            if env_tp_size.isdigit():
                tp_size = int(env_tp_size)
                break
    if tp_size is None:
        num_attention_heads = read_num_attention_heads_from_raw_config(model_path)
        if num_attention_heads is not None:
            valid_tp = [
                tp for tp in range(min(gpu_count, int(num_attention_heads)), 0, -1)
                if int(num_attention_heads) % tp == 0
            ]
            tp_size = valid_tp[0] if valid_tp else 1
        else:
            tp_size = 1 if gpu_count <= 1 else 2 if gpu_count <= 3 else 4 if gpu_count <= 7 else 8
    return max(1, min(int(tp_size), max(1, gpu_count)))



def load_hf_model(model_cls, model_path: str, *, is_omni: bool, trust_remote_code: bool = False):
    load_kwargs = {'device_map': 'auto'}
    if is_omni:
        load_kwargs['dtype'] = 'auto'
    else:
        load_kwargs['torch_dtype'] = 'auto'
        if trust_remote_code:
            load_kwargs['trust_remote_code'] = True
    try:
        return model_cls.from_pretrained(model_path, attn_implementation='flash_attention_2', **load_kwargs)
    except Exception as err:
        logging.warning(
            f'Falling back to default attention for {model_path}: {type(err).__name__}: {err}'
        )
        return model_cls.from_pretrained(model_path, **load_kwargs)


def process_vision_info_compat(process_vision_info, messages):
    try:
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        video_metadatas = None
        if videos is not None and len(videos) > 0 and isinstance(videos[0], tuple) and len(videos[0]) == 2:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        return images, videos, video_kwargs, video_metadatas
    except TypeError as err:
        if 'image_patch_size' not in str(err) and 'return_video_metadata' not in str(err):
            raise
        outputs = process_vision_info(messages, return_video_kwargs=True)
        if len(outputs) == 3:
            images, videos, video_kwargs = outputs
        else:
            images, videos = outputs
            video_kwargs = None
        if videos is None and isinstance(video_kwargs, dict):
            if all((v in (None, [], (), {}) for v in video_kwargs.values())):
                video_kwargs = None
        return images, videos, video_kwargs, None

def apply_chat_template_compat(processor, messages, **kwargs):
    return processor.apply_chat_template(messages, **kwargs)


def apply_chat_template_nothink(processor, messages, **kwargs):
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def apply_chat_template_think(processor, messages, **kwargs):
    try:
        return processor.apply_chat_template(messages, enable_thinking=True, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


class Qwen35VLChatReplay(Qwen3VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens: int = 32768,
        top_p: float = 0.95,
        top_k: int = 20,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        use_custom_prompt: bool = False,
        system_prompt: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
        use_audio_in_video: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        self.FRAME_FACTOR = 2
        self.use_audio_in_video = use_audio_in_video

        assert model_path is not None
        self.model_path = model_path

        from transformers import AutoProcessor, AutoModelForImageTextToText
        if listinstr(['omni'], model_path.lower()):
            try:
                from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
            except Exception as err:
                logging.critical('pip install git+https://github.com/huggingface/transformers')
                raise err
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
            model_cls = Qwen3OmniMoeForConditionalGeneration
        else:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            model_cls = AutoModelForImageTextToText

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems else -1
        assert max_gpu_mem > 0

        self.use_vllm = kwargs.get('use_vllm', True)
        env_limit_mm = os.environ.get('REPLAY_LIMIT_MM_PER_PROMPT', '').strip()
        if env_limit_mm.isdigit():
            self.limit_mm_per_prompt = max(1, int(env_limit_mm))
        else:
            self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

        if self.use_vllm:
            from vllm import LLM
            gpu_count = torch.cuda.device_count()
            env_max_model_len = (
                os.environ.get('QWEN35_VLLM_MAX_MODEL_LEN', '').strip()
                or os.environ.get('VLLM_MAX_MODEL_LEN', '').strip()
            )
            env_max_num_seqs = (
                os.environ.get('QWEN35_VLLM_MAX_NUM_SEQS', '').strip()
                or os.environ.get('VLLM_MAX_NUM_SEQS', '').strip()
            )
            tp_size = resolve_vllm_tp_size(
                self.model_path,
                gpu_count,
                explicit_tp_size=kwargs.get('tensor_parallel_size', None),
            )
            max_model_len = kwargs.get('max_model_len', None)
            if max_model_len is None:
                max_model_len = int(env_max_model_len) if env_max_model_len.isdigit() else 32768
            max_num_seqs = int(env_max_num_seqs) if env_max_num_seqs.isdigit() else 2
            env_enforce_eager = (
                os.environ.get('QWEN35_VLLM_ENFORCE_EAGER', '').strip().lower()
                or os.environ.get('VLLM_ENFORCE_EAGER', '').strip().lower()
            )
            enforce_eager = env_enforce_eager in {'1', 'true', 'yes', 'on'}
            enable_expert_parallel = is_moe_model(self.model_path)
            limit_mm = {'image': self.limit_mm_per_prompt}
            if listinstr(['omni'], self.model_path.lower()):
                limit_mm = {'image': 3, 'video': 3, 'audio': 3}
                os.environ['VLLM_USE_V1'] = '0'
            try:
                self.llm = LLM(
                    model=self.model_path,
                    max_num_seqs=max_num_seqs,
                    max_model_len=max_model_len,
                    limit_mm_per_prompt=limit_mm,
                    tensor_parallel_size=tp_size,
                    enable_expert_parallel=enable_expert_parallel,
                    gpu_memory_utilization=kwargs.get('gpu_utils', 0.9),
                    trust_remote_code=True,
                    enforce_eager=enforce_eager,
                    seed=0,
                )
            except Exception as err:
                print(f'[Qwen35VLChatReplay] vLLM init failed, fallback to transformers: {type(err).__name__}: {err}', flush=True)
                self.use_vllm = False
                if listinstr(['omni'], model_path.lower()):
                    self.model = load_hf_model(model_cls, model_path, is_omni=True)
                else:
                    self.model = load_hf_model(model_cls, model_path, is_omni=False, trust_remote_code=True)
                self.model.eval()
        else:
            if listinstr(['omni'], model_path.lower()):
                self.model = load_hf_model(model_cls, model_path, is_omni=True)
            else:
                self.model = load_hf_model(model_cls, model_path, is_omni=False, trust_remote_code=True)
            self.model.eval()

        torch.cuda.empty_cache()

        self._stage_debug_enabled = os.environ.get('REPLAY_STAGE_DEBUG', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
        self._stage_debug_max_samples = int(os.environ.get('REPLAY_STAGE_DEBUG_SAMPLES', '3'))
        self._stage_debug_seen_samples = 0
        self._stage_debug_active = False
        self._replay_dump_dir = os.environ.get('REPLAY_DUMP_DIR', '').strip()
        self._replay_dump_file = None
        self._replay_dump_max_chars = int(os.environ.get('REPLAY_DUMP_MAX_CHARS', '0'))
        self._prompt_audit_enabled = os.environ.get('REPLAY_PROMPT_AUDIT', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
        self._prompt_audit_print = os.environ.get('REPLAY_PROMPT_AUDIT_PRINT', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
        if self._replay_dump_dir:
            os.makedirs(self._replay_dump_dir, exist_ok=True)
            self._replay_dump_file = os.path.join(self._replay_dump_dir, f'{self.__class__.__name__}.jsonl')
            print(f'[replay-dump] enabled. Writing to {self._replay_dump_file}', flush=True)

        self.replay_cfg = read_replay_config_from_env()
        self.prompt_template_cfg = read_prompt_template_config_from_env()
        self.template_on_last_replay_text = os.environ.get('REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT', '0').strip().lower() in {
            '1', 'true', 'yes', 'on'
        }
        self.safe_fallback_enabled = os.environ.get('REPLAY_SAFE_FALLBACK', '1').strip().lower() in {
            '1', 'true', 'yes', 'on'
        }
        self.safe_fallback_truncate_chars = max(512, int(os.environ.get('REPLAY_SAFE_TRUNCATE_CHARS', '6000')))
        print(f'[Qwen35VLChatReplay] replay_cfg={self.replay_cfg}', flush=True)
        print(f'[Qwen35VLChatReplay] prompt_template_cfg={self.prompt_template_cfg}', flush=True)
        print(f'[Qwen35VLChatReplay] template_on_last_replay_text={self.template_on_last_replay_text}', flush=True)

    def _prefer_direct_answer_mode(self) -> bool:
        return self.prompt_template_cfg.get('name') == 'directly_answer'

    def _resolve_generation_profile(self, dataset: str | None = None):
        return resolve_qwen35_generation_profile(
            dataset,
            force_no_thinking=self._prefer_direct_answer_mode(),
            max_new_tokens=self.max_new_tokens,
            thinking_temperature=self.temperature,
            thinking_top_p=self.top_p,
            thinking_top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
        )

    def _build_transformers_generate_kwargs(self, dataset: str | None = None):
        profile = self._resolve_generation_profile(dataset)
        return dict(
            max_new_tokens=profile['max_new_tokens'],
            top_p=profile['top_p'],
            top_k=profile['top_k'],
            min_p=profile['min_p'],
            temperature=profile['temperature'],
            repetition_penalty=profile['repetition_penalty'],
        )

    def _apply_chat_template(self, messages, dataset: str | None = None, **kwargs):
        profile = self._resolve_generation_profile(dataset)
        if profile['enable_thinking']:
            return apply_chat_template_think(self.processor, messages, **kwargs)
        return apply_chat_template_nothink(self.processor, messages, **kwargs)

    def _begin_stage_debug_sample(self):
        if self._stage_debug_enabled and self._stage_debug_seen_samples < self._stage_debug_max_samples:
            self._stage_debug_seen_samples += 1
            self._stage_debug_active = True
        else:
            self._stage_debug_active = False

    def _stage_debug(self, stage: str, payload: dict):
        if not self._stage_debug_active:
            return
        info = {'stage': stage, 'model': self.__class__.__name__}
        info.update(payload)
        try:
            print('[STAGE_DEBUG] ' + json.dumps(info, ensure_ascii=False), flush=True)
        except Exception:
            print(f'[STAGE_DEBUG] stage={stage} model={self.__class__.__name__}', flush=True)

    def _clip_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        max_chars = self._replay_dump_max_chars
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars] + f'\n...[TRUNCATED {len(text) - max_chars} chars]'
        return text

    @staticmethod
    def _has_boxed_instruction(text: str) -> bool:
        if not isinstance(text, str):
            return False
        lowered = text.lower()
        return ('\boxed{' in text) or ('boxed{<answer>}' in lowered)

    @staticmethod
    def _has_one_line_instruction(text: str) -> bool:
        if not isinstance(text, str):
            return False
        return 'return exactly one line in this format' in text.lower()

    def _collect_text_blocks(self, content: list[dict]) -> list[dict]:
        blocks = []
        for idx, item in enumerate(content):
            if not isinstance(item, dict) or item.get('type') != 'text':
                continue
            text = item.get('text', '')
            blocks.append({
                'index': idx,
                'chars': len(text),
                'has_boxed_instruction': self._has_boxed_instruction(text),
                'has_one_line_instruction': self._has_one_line_instruction(text),
                'text': self._clip_text(text),
            })
        return blocks

    def _write_replay_dump(self, record: dict):
        if not self._replay_dump_file:
            return
        payload = {'ts': time.time(), 'model_class': self.__class__.__name__}
        payload.update(record)
        with open(self._replay_dump_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')

    def _apply_prompt_template_to_content(self, content: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        templated = apply_prompt_template_to_content(content, self.prompt_template_cfg, dataset=dataset)
        before_blocks = self._collect_text_blocks(content)
        after_blocks = self._collect_text_blocks(templated)
        audit_info = {
            'template_name': self.prompt_template_cfg.get('name'),
            'template_source': self.prompt_template_cfg.get('source'),
            'before_text_blocks': before_blocks,
            'after_text_blocks': after_blocks,
            'before_any_boxed_instruction': any(x.get('has_boxed_instruction') for x in before_blocks),
            'after_any_boxed_instruction': any(x.get('has_boxed_instruction') for x in after_blocks),
            'before_any_one_line_instruction': any(x.get('has_one_line_instruction') for x in before_blocks),
            'after_any_one_line_instruction': any(x.get('has_one_line_instruction') for x in after_blocks),
        }
        self._stage_debug('prompt_template_before_after', audit_info)
        if self._prompt_audit_enabled:
            self._write_replay_dump({'phase': 'prompt_template_before_after', **audit_info})
            if self._prompt_audit_print:
                print('[PROMPT_AUDIT] ' + json.dumps({
                    'template': self.prompt_template_cfg.get('name'),
                    'before_boxed': audit_info['before_any_boxed_instruction'],
                    'after_boxed': audit_info['after_any_boxed_instruction'],
                    'before_one_line': audit_info['before_any_one_line_instruction'],
                    'after_one_line': audit_info['after_any_one_line_instruction'],
                }, ensure_ascii=False), flush=True)
        return templated

    def _apply_replay_to_content(self, content: list[dict[str, str]]) -> list[dict[str, str]]:
        replayed = apply_replay(
            content,
            mode=self.replay_cfg['mode'],
            repeat_times=self.replay_cfg['repeat_times'],
            image_copy_mode=self.replay_cfg['image_copy_mode'],
        )
        self._stage_debug('replay_before_after', {
            'replay_mode': self.replay_cfg['mode'],
            'before_preview': str(content)[:1200],
            'after_preview': str(replayed)[:1200],
        })
        maybe_debug_print_replay(
            enabled=self.replay_cfg['debug'],
            mode=self.replay_cfg['mode'],
            before=content,
            after=replayed,
            tag=self.__class__.__name__,
        )
        return replayed

    def _apply_template_replay_pipeline(self, content: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        replay_mode = canonicalize_replay_mode(self.replay_cfg.get('mode', 'image_text'))
        if self.template_on_last_replay_text and not is_noop_replay_mode(replay_mode):
            replay_source = content
            if self.prompt_template_cfg.get('name') == 'directly_answer':
                replay_source = strip_prompt_template_from_content_for_direct_answer(content, dataset=dataset, text_key='text')
            replayed = self._apply_replay_to_content(replay_source)
            return self._apply_prompt_template_to_content(replayed, dataset=dataset)
        templated = self._apply_prompt_template_to_content(content, dataset=dataset)
        return self._apply_replay_to_content(templated)

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 32 * 32
                    warnings.warn(f'OCRBench dataset uses custom min_pixels={item["min_pixels"]}')
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['min_pixels', 'max_pixels', 'total_pixels', 'resized_height', 'resized_width']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
            elif s['type'] == 'video':
                value = s['value']
                if isinstance(value, list):
                    item = {'type': 'video', 'video': [ensure_image_url(v) for v in value]}
                else:
                    item = {'type': 'video', 'video': ensure_video_url(value)}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['resized_height', 'resized_width', 'fps', 'nframes', 'sample_fps']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
                if not isinstance(value, list):
                    if self.fps is not None and 'fps' not in item:
                        item['fps'] = self.fps
                    elif self.nframe is not None and 'nframes' not in item:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        item['nframes'] = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR if frame_count < self.nframe else self.nframe
            elif s['type'] == 'audio':
                item = {'type': 'audio', 'audio': s['value']}
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f'Invalid message type: {s["type"]}, {s}')
            content.append(item)
        return self._apply_template_replay_pipeline(content, dataset=dataset)

    def _post_process_response(self, response: str) -> str:
        if self.post_process:
            resp = response.split('\boxed{')[-1]
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

    def generate_inner_transformers(self, message, dataset=None):
        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical('Please install it via \'pip install qwen-omni-utils[decord]\'')
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical('Please install it via \'pip install qwen-vl-utils\'')
                raise err
        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')
        if is_omni:
            text = self._apply_chat_template(messages, dataset=dataset, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors='pt',
                padding=True,
                use_audio_in_video=self.use_audio_in_video,
            )
        else:
            text = self._apply_chat_template(messages, dataset=dataset, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs, video_metadatas = process_vision_info_compat(process_vision_info, messages)
            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                do_resize=False,
                return_tensors='pt',
                **(video_kwargs or {}),
            )
        try:
            inputs = inputs.to(self.model.device)
            if hasattr(self.model, 'dtype'):
                inputs = inputs.to(self.model.dtype)
        except Exception:
            inputs = inputs.to('cuda')
        if is_omni:
            try:
                text_ids, _ = self.model.generate(
                    **inputs,
                    return_audio=False,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=self.use_audio_in_video,
                )
            except TypeError:
                text_ids, _ = self.model.generate(
                    **inputs,
                    return_audio=False,
                    use_audio_in_video=self.use_audio_in_video,
                )
            response = self.processor.batch_decode(
                text_ids.sequences[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        else:
            generated_ids = self.model.generate(**inputs, **self._build_transformers_generate_kwargs(dataset))
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
            response = self.processor.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        return self._post_process_response(response)

    def _build_vllm_sampling_params(self, dataset: str | None = None):
        from vllm import SamplingParams
        profile = self._resolve_generation_profile(dataset)
        return SamplingParams(
            temperature=profile['temperature'],
            max_tokens=profile['max_new_tokens'],
            top_p=profile['top_p'],
            top_k=profile['top_k'],
            min_p=profile['min_p'],
            repetition_penalty=profile['repetition_penalty'],
            presence_penalty=profile['presence_penalty'],
            stop_token_ids=None,
        )

    def _build_vllm_request(self, message, dataset=None):
        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, 'pip install qwen-vl-utils'")
                raise err
        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')
        text = self._apply_chat_template(messages, dataset=dataset, tokenize=False, add_generation_prompt=True)
        video_kwargs = None
        if is_omni:
            audios, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
        else:
            image_inputs, video_inputs, video_kwargs, _ = process_vision_info_compat(process_vision_info, messages)
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        if is_omni and 'audios' in locals() and audios is not None:
            mm_data['audio'] = audios
        req = {'prompt': text}
        if mm_data:
            req['multi_modal_data'] = mm_data
        if is_omni:
            req['mm_processor_kwargs'] = {'use_audio_in_video': self.use_audio_in_video}
        elif video_kwargs is not None:
            req['mm_processor_kwargs'] = video_kwargs
        return req

    def _extract_vllm_output_text(self, output):
        if getattr(output, 'outputs', None):
            return output.outputs[0].text
        return ''

    def _generate_inner_vllm_once(self, message, dataset=None):
        sampling_params = self._build_vllm_sampling_params(dataset)
        req = self._build_vllm_request(message, dataset=dataset)
        outputs = self.llm.generate([req], sampling_params=sampling_params)
        generated_text = self._extract_vllm_output_text(outputs[0])
        return self._post_process_response(generated_text)

    def _generate_batch_inner_vllm(self, messages, dataset=None):
        sampling_params = self._build_vllm_sampling_params()
        reqs = [self._build_vllm_request(message, dataset=dataset) for message in messages]
        outputs = self.llm.generate(reqs, sampling_params=sampling_params)
        return [self._post_process_response(self._extract_vllm_output_text(output)) for output in outputs]

    def _run_with_replay_cfg(self, message, dataset, *, replay_mode: str, repeat_times: int) -> str:
        old_cfg = dict(self.replay_cfg)
        self.replay_cfg = dict(self.replay_cfg)
        self.replay_cfg['mode'] = replay_mode
        self.replay_cfg['repeat_times'] = repeat_times
        try:
            return self._generate_inner_vllm_once(message, dataset=dataset)
        finally:
            self.replay_cfg = old_cfg

    def _truncate_message_text(self, message: list[dict[str, str]]) -> list[dict[str, str]]:
        out = []
        max_chars = self.safe_fallback_truncate_chars
        for item in message:
            if isinstance(item, dict) and item.get('type') == 'text':
                new_item = dict(item)
                txt = item.get('value', '')
                if isinstance(txt, str) and len(txt) > max_chars:
                    new_item['value'] = txt[:max_chars] + f'\n...[TRUNCATED {len(txt) - max_chars} chars]'
                out.append(new_item)
            else:
                out.append(item)
        return out

    def generate_inner_vllm(self, message, dataset=None):
        try:
            return self._generate_inner_vllm_once(message, dataset=dataset)
        except Exception as first_err:
            if not self.safe_fallback_enabled:
                raise
            replay_mode = canonicalize_replay_mode(self.replay_cfg.get('mode', 'image_text'))
            print(f'[Qwen35VLChatReplay][safe-fallback] primary generation failed: {type(first_err).__name__}: {first_err}', flush=True)
            if not is_noop_replay_mode(replay_mode):
                try:
                    out = self._run_with_replay_cfg(message, dataset, replay_mode='image_text', repeat_times=1)
                    self._write_replay_dump({
                        'phase': 'safe_fallback',
                        'strategy': 'disable_replay',
                        'dataset': str(dataset) if dataset is not None else None,
                        'original_replay_mode': replay_mode,
                        'error_type': type(first_err).__name__,
                        'error': self._clip_text(str(first_err)),
                    })
                    return out
                except Exception as disable_err:
                    print(f'[Qwen35VLChatReplay][safe-fallback] disable_replay failed: {type(disable_err).__name__}: {disable_err}', flush=True)
            trimmed_message = self._truncate_message_text(message)
            out = self._run_with_replay_cfg(trimmed_message, dataset, replay_mode='image_text', repeat_times=1)
            self._write_replay_dump({
                'phase': 'safe_fallback',
                'strategy': 'disable_replay_and_truncate_text',
                'dataset': str(dataset) if dataset is not None else None,
                'original_replay_mode': replay_mode,
                'truncate_chars': self.safe_fallback_truncate_chars,
                'error_type': type(first_err).__name__,
                'error': self._clip_text(str(first_err)),
            })
            return out

    def generate_inner(self, message, dataset=None):
        self._begin_stage_debug_sample()
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        return self.generate_inner_transformers(message, dataset=dataset)

    def generate_batch_inner(self, messages, dataset=None):
        if isinstance(messages, list) and messages and isinstance(messages[0], list):
            self._begin_stage_debug_sample()
            if self.use_vllm:
                try:
                    return self._generate_batch_inner_vllm(messages, dataset=dataset)
                except Exception as batch_err:
                    print(
                        f'[Qwen35VLChatReplay][batch-fallback] batched vLLM generation failed: '
                        f'{type(batch_err).__name__}: {batch_err}',
                        flush=True,
                    )
                    return [self.generate_inner_vllm(msg, dataset=dataset) for msg in messages]
            return [self.generate_inner(msg, dataset=dataset) for msg in messages]
        return [self.generate_inner(messages, dataset=dataset)]
