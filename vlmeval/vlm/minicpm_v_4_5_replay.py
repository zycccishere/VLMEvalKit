import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from .base import BaseModel
from .qwen2_vl.replay_prompt_template import (
    read_prompt_template_config_from_env,
    render_prompt_with_template,
    strip_prompt_template_from_content_for_direct_answer,
)
from .replay_policy import (
    REPLAY_IMAGE_TEXT_FILLER_TEXT,
    apply_replay,
    canonicalize_replay_mode,
    is_noop_replay_mode,
    maybe_debug_print_replay,
    read_replay_config_from_env,
)
from .replay_text_filler import (
    SPACE_FILLER_LITERAL,
    TextFillerAudit,
    build_space_filler,
    count_filler_tokens,
    tokenize_length,
)
from ..smp import *


def _env_truthy(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    raw = os.environ.get(name, "").strip()
    if raw.isdigit():
        return int(raw)
    return default


def _dataset_type(name):
    from ..dataset import DATASET_TYPE

    return DATASET_TYPE(name)


class MiniCPM_V_4_5(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path="openbmb/MiniCPM-V-4_5", **kwargs):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        assert model_path is not None
        self.model_path = model_path
        print(f"load from path {self.model_path}")
        self.kwargs = dict(kwargs)
        self.use_vllm = self.kwargs.pop("use_vllm", _env_truthy("MINICPM45_USE_VLLM", False))
        self.vllm_tensor_parallel_size = self.kwargs.pop("tensor_parallel_size", None)
        self.vllm_max_model_len = _env_int("MINICPM45_VLLM_MAX_MODEL_LEN", 32768)
        self.vllm_max_num_seqs = _env_int(
            "MINICPM45_VLLM_MAX_NUM_SEQS",
            _env_int("INFER_BATCH_SIZE", 4),
        )
        self.vllm_max_images = _env_int("MINICPM45_VLLM_MAX_IMAGES", 8)
        self.vllm_gpu_memory_utilization = float(
            os.environ.get("MINICPM45_VLLM_GPU_MEMORY_UTILIZATION", "0.85")
        )
        self.generate_overrides = {}
        for key in (
            "enable_thinking",
            "max_new_tokens",
            "sampling",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "presence_penalty",
            "num_beams",
            "no_repeat_ngram_size",
        ):
            if key in self.kwargs:
                self.generate_overrides[key] = self.kwargs.pop(key)

        self.model = None
        self.llm = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        if self.use_vllm:
            from vllm import LLM
            from vllm.transformers_utils.chat_templates.registry import CHAT_TEMPLATES_DIR

            tp_size = self._resolve_vllm_tp_size()
            self.vllm_chat_template = self._build_vllm_chat_template(
                Path(CHAT_TEMPLATES_DIR) / "template_minicpmv45.jinja"
            )
            self.llm = LLM(
                model=self.model_path,
                trust_remote_code=True,
                tensor_parallel_size=tp_size,
                max_model_len=max(1024, self.vllm_max_model_len),
                max_num_seqs=max(1, self.vllm_max_num_seqs),
                limit_mm_per_prompt={"image": max(1, self.vllm_max_images)},
                gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                chat_template=self.vllm_chat_template,
            )
            print(
                f"[MiniCPM_V_4_5] using vLLM tp={tp_size} "
                f"max_model_len={self.vllm_max_model_len} max_num_seqs={self.vllm_max_num_seqs}",
                flush=True,
            )
        else:
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = self.model.to(dtype=torch.bfloat16)
            self.model.eval().cuda()
        torch.cuda.empty_cache()

        self.num_beams = 3
        self.max_new_tokens = _env_int("MINICPM45_MAX_NEW_TOKENS", 16384)
        self.options_suffix_prompt = "\nAnswer with the option's letter from the given choices directly."
        self.wo_options_system_prompt = "Carefully read the following question. Answer the question directly."
        self.detail_system_prompt = "Answer this question in detail."
        self.vqa_prompt = "Answer the question using a single word or phrase."
        self.multi_choice_cot_prompt = (
            "Carefully read the following multichoice question, solve it step "
            'by step and finally pick the option associated with the correct '
            'answer in the format of "Answer: selected option\n\n'
        )
        self.short_ans_cot_prompt = (
            "Read the following question carefully, solve it step by step, and "
            'then output the final answer in the format of "Answer: single number '
            'or single word or phrase".\n\n'
        )
        self.ocrbench_cot_prompt = "Carefully observe the image and answer the OCR-related questions below. \n\n"

        self._original_chat_template = self.tokenizer.chat_template
        self._no_think_chat_template = self._strip_empty_think_stub(self._original_chat_template)
        self._long_cot_chat_template = self._original_chat_template
        self.debug_io_enabled = os.environ.get("MINICPM_DEBUG_IO", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.debug_io_every = max(1, int(os.environ.get("MINICPM_DEBUG_IO_EVERY", "50")))
        self.debug_io_max_text_chars = max(200, int(os.environ.get("MINICPM_DEBUG_IO_MAX_TEXT_CHARS", "4000")))
        self.debug_io_max_output_chars = max(200, int(os.environ.get("MINICPM_DEBUG_IO_MAX_OUTPUT_CHARS", "4000")))
        self.debug_io_max_items = max(1, int(os.environ.get("MINICPM_DEBUG_IO_MAX_ITEMS", "32")))
        self.debug_io_dir = os.environ.get("MINICPM_DEBUG_IO_DIR", "").strip()
        self.debug_io_file = None
        if self.debug_io_dir:
            os.makedirs(self.debug_io_dir, exist_ok=True)
            self.debug_io_file = os.path.join(self.debug_io_dir, f"{self.__class__.__name__}.jsonl")
        self._debug_io_counter = 0
        self._warned_vllm_beam_fallback = False
        self._text_filler_audit = TextFillerAudit(self.__class__.__name__)
        self._packaged_vllm_chat_template = None

    def _minicpm_image_placeholder_pattern(self) -> str:
        return "(<image>./</image>)"

    def _packaged_chat_template_str(self) -> str:
        if self._packaged_vllm_chat_template is None:
            from vllm.transformers_utils.chat_templates.registry import CHAT_TEMPLATES_DIR

            path = Path(CHAT_TEMPLATES_DIR) / "template_minicpmv45.jinja"
            self._packaged_vllm_chat_template = Path(path).read_text(encoding="utf-8")
        return self._packaged_vllm_chat_template

    def _minicpm_visual_core_tokens(self, message):
        """Core visual token count |V| for a message, using the same slicing
        the vLLM engine applies.

        vLLM resolves ``max_slice_nums`` from the checkpoint image processor
        (MiniCPM-o 4_5: 9 in preprocessor_config.json even though the model
        config's slice_config says 1), and every sub-image (thumbnail plus
        slices) contributes ``image_feature_size`` resampler tokens.
        """
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            raise RuntimeError("MiniCPM processor exposes no image_processor")
        feature_size = int(getattr(image_processor, "image_feature_size", 0) or 0)
        if feature_size <= 0:
            raise RuntimeError("MiniCPM image processor lacks image_feature_size")
        max_slice_nums = getattr(image_processor, "max_slice_nums", None)
        details = []
        total = 0
        for item in message:
            if not (isinstance(item, dict) and item.get("type") == "image"):
                continue
            with Image.open(str(item["value"])) as handle:
                width, height = handle.size
            grid = image_processor.get_sliced_grid((width, height), max_slice_nums)
            slices = 0 if grid is None else int(grid[0]) * int(grid[1])
            core = (slices + 1) * feature_size
            total += core
            details.append(
                {
                    "size": [width, height],
                    "grid": None if grid is None else [int(grid[0]), int(grid[1])],
                    "slices": slices,
                    "core": core,
                }
            )
        return total, details

    def _render_vllm_chat_string(self, content_str, dataset=None):
        """Mirror the chat-template branch ``_generate_batch_inner_vllm``
        feeds to ``llm.chat`` so filler audits predict the engine prompt."""
        _, chat_template_kwargs = self._build_vllm_sampling(dataset=dataset)
        if chat_template_kwargs and chat_template_kwargs.get("enable_thinking") is False:
            template = self._original_chat_template
        else:
            template = self._packaged_chat_template_str()
        kwargs = dict(tokenize=False, add_generation_prompt=True, chat_template=template)
        if chat_template_kwargs:
            kwargs.update(chat_template_kwargs)
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content_str}], **kwargs
        )

    def _expand_image_placeholders(self, prompt_str, image_items):
        image_processor = getattr(self.processor, "image_processor", None)
        pattern = self._minicpm_image_placeholder_pattern()
        parts = []
        cursor = 0
        for item in image_items:
            with Image.open(str(item["value"])) as handle:
                width, height = handle.size
            placeholder = image_processor.get_slice_image_placeholder((width, height))
            idx = prompt_str.find(pattern, cursor)
            if idx < 0:
                raise RuntimeError("fewer image placeholders than images while expanding")
            parts.append(prompt_str[cursor:idx])
            parts.append(placeholder)
            cursor = idx + len(pattern)
        parts.append(prompt_str[cursor:])
        if prompt_str.find(pattern, cursor) >= 0:
            raise RuntimeError("more image placeholders than images while expanding")
        return "".join(parts)

    def _predict_engine_prompt_len(self, message, dataset=None):
        """Local replica of the vendored MiniCPMV processor's token count for
        the prompt the engine will build for ``message``."""
        texts = [
            str(item["value"])
            for item in message
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        image_items = [
            item
            for item in message
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        content = "\n".join(texts)
        if image_items:
            content = self._minicpm_image_placeholder_pattern() + "\n" + content
        prompt_str = self._render_vllm_chat_string(content, dataset=dataset)
        final_text = (
            self._expand_image_placeholders(prompt_str, image_items)
            if image_items
            else prompt_str
        )
        # vllm/transformers_utils/processors/minicpmv.py _convert() prepends
        # the bos id manually (add_bos_token=False on this tokenizer).
        return len(self.tokenizer.encode(final_text)) + 1

    def _minicpm_unk_token_id(self) -> int:
        image_processor = getattr(self.processor, "image_processor", None)
        literal = getattr(image_processor, "unk_token", None) or getattr(
            self.tokenizer, "unk_token", "<unk>"
        )
        token_id = int(self.tokenizer.convert_tokens_to_ids(str(literal)))
        if token_id is None or token_id < 0:
            raise RuntimeError(f"cannot resolve MiniCPM image feature token {literal!r}")
        return token_id

    def _audit_text_filler_engine_outputs(self, messages, outputs, dataset=None):
        if not any(
            isinstance(item, dict) and item.get("text_filler_meta")
            for message in messages
            for item in message
        ):
            return
        unk_id = self._minicpm_unk_token_id()
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
            unk_count = sum(1 for token_id in ids if token_id == unk_id)
            baseline_prediction = int(meta["baseline_engine_len_predicted"])
            filler_prediction = int(meta["filler_engine_len_predicted"])
            visual_core = int(meta["visual_core_tokens"])
            filler_engine = engine_len - baseline_prediction
            record = {
                **meta,
                "engine_prompt_len": engine_len,
                "engine_prompt_len_predicted": filler_prediction,
                "engine_image_core_count": unk_count,
                "filler_tokens_engine": filler_engine,
            }
            if engine_len != filler_prediction:
                raise RuntimeError(
                    f"[text-filler] ENGINE AUDIT MISMATCH ({self.__class__.__name__}, "
                    f"dataset={dataset}): engine prompt len {engine_len} != predicted "
                    f"{filler_prediction}; record={record}"
                )
            if unk_count != visual_core:
                raise RuntimeError(
                    f"[text-filler] ENGINE AUDIT MISMATCH ({self.__class__.__name__}, "
                    f"dataset={dataset}): engine image core count {unk_count} != "
                    f"visual core {visual_core}; record={record}"
                )
            self._text_filler_audit.log_engine(record)

    def _resolve_vllm_tp_size(self):
        gpu_count = max(1, torch.cuda.device_count())
        tp_size = self.vllm_tensor_parallel_size
        if tp_size is None:
            env_tp = os.environ.get("MINICPM45_VLLM_TP_SIZE", "").strip()
            tp_size = int(env_tp) if env_tp.isdigit() else 1
        tp_size = max(1, min(int(tp_size), gpu_count))
        return tp_size

    def _strip_empty_think_stub(self, chat_template):
        pattern = (
            r"\s*\{%- if enable_thinking is defined and enable_thinking is false %\}\s*"
            r"\{\{- '<think>\\n\\n</think>\\n\\n' \}\}\s*"
            r"\{%- endif %\}"
        )
        return re.sub(pattern, "", str(chat_template), count=1)

    def _build_vllm_chat_template(self, packaged_template_path: Path) -> str:
        return str(packaged_template_path)

    def use_custom_prompt(self, dataset=None):
        if dataset is None:
            return False
        return listinstr(["MCQ", "VQA", "Y/N"], _dataset_type(dataset))

    def _prefer_direct_answer_mode(self):
        template_name = (
            os.environ.get("REPLAY_PROMPT_TEMPLATE_NAME")
            or os.environ.get("PROMPT_TEMPLATE_NAME")
            or ""
        ).strip().lower()
        force_no_thinking = os.environ.get("MINICPM_FORCE_NO_THINKING", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return force_no_thinking and template_name == "directly_answer"

    def _reasoning_mode_override(self):
        mode = os.environ.get("MINICPM45_REASONING_MODE", "").strip().lower().replace("-", "_")
        if mode in {"no_thinking", "no_reasoning", "regular_cot", "regular_reasoning", "long_cot"}:
            return mode
        return ""

    def _use_general_no_thinking_dataset(self, dataset=None):
        override = self._reasoning_mode_override()
        if override in {"regular_cot", "regular_reasoning", "long_cot"}:
            return False
        if override in {"no_thinking", "no_reasoning"}:
            return True
        if dataset is None:
            return False
        return listinstr(["AI2D", "AI2D_TEST", "SEEDBench2_Plus"], dataset)

    def _should_disable_thinking(self, dataset=None):
        override = self._reasoning_mode_override()
        if override in {"no_thinking", "no_reasoning"}:
            return True
        if override in {"regular_cot", "regular_reasoning", "long_cot"}:
            return False
        return self._prefer_direct_answer_mode() or self._use_general_no_thinking_dataset(dataset)

    def _select_hf_chat_template(self, dataset=None):
        if self.use_long_cot(dataset):
            return self._long_cot_chat_template
        return self._original_chat_template

    def use_long_cot(self, dataset=None):
        override = self._reasoning_mode_override()
        if override == "long_cot":
            return True
        if override in {"no_thinking", "no_reasoning", "regular_cot", "regular_reasoning"}:
            return False
        if self._use_general_no_thinking_dataset(dataset):
            return False
        if self._prefer_direct_answer_mode():
            return False
        if dataset is None:
            return False
        return listinstr(
            [
                "MMMU",
                "MathVista",
                "MMVet",
                "MMBench",
                "HallusionBench",
                "MMStar",
                "MathVision",
                "MathVerse_MINI",
                "MathVerse_MINI_Vision_Only",
                "DynaMath",
                "LogicVista",
                "VisualPuzzles",
                "WeMath",
            ],
            dataset,
        )

    def use_cot(self, dataset=None):
        override = self._reasoning_mode_override()
        if override in {"regular_cot", "regular_reasoning", "long_cot"}:
            return True
        if override in {"no_thinking", "no_reasoning"}:
            return False
        if self._use_general_no_thinking_dataset(dataset):
            return False
        if self._prefer_direct_answer_mode():
            return False
        if dataset is None:
            return False
        return listinstr(
            [
                "MMMU",
                "MathVista",
                "MMBench",
                "HallusionBench",
                "MMStar",
                "OCRBench",
                "ChartQA",
                "MathVision",
                "MathVerse_MINI",
                "MathVerse_MINI_Vision_Only",
                "DynaMath",
                "LogicVista",
                "VisualPuzzles",
                "WeMath",
            ],
            dataset,
        )

    def use_upsize(self, dataset=None):
        # Evaluation should preserve the benchmark image as provided. The
        # previous MiniCPM 4.5 wrapper upsampled selected datasets to roughly
        # 1344x1344 with a random width, which made the runner apply a
        # model-specific test-time augmentation. Keep the old policy disabled
        # for both HF and vLLM paths.
        return False

    def build_prompt(self, line, dataset=None):
        self.tokenizer.chat_template = self._select_hf_chat_template(dataset)

        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
        system_prompt, prompt = "", ""
        question = line["question"]

        if not self.use_cot(dataset):
            if _dataset_type(dataset) == "MCQ":
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = "Options:\n"
                for key, item in options.items():
                    options_prompt += f"{key}. {item}\n"
                hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
                if hint is not None:
                    prompt += f"Hint: {hint}\n"
                prompt += f"Question: {question}\n"
                if len(options):
                    prompt += options_prompt
                    prompt += self.options_suffix_prompt
                else:
                    system_prompt = self.wo_options_system_prompt

                if dataset is not None and "MMMU" in dataset and system_prompt:
                    prompt = system_prompt + "\n" + prompt
                    system_prompt = ""
            elif dataset is not None and listinstr(["HallusionBench"], dataset):
                prompt = question + " Yes or No?"
            elif dataset is not None and listinstr(["OCRBench"], dataset):
                system_prompt = self.vqa_prompt
                prompt = question
            elif _dataset_type(dataset) == "VQA":
                if dataset is not None and listinstr(["LLaVABench"], dataset):
                    system_prompt = ""
                elif dataset is not None and listinstr(["MMVet"], dataset):
                    system_prompt = self.detail_system_prompt
                else:
                    system_prompt = self.vqa_prompt
                prompt = question
            else:
                prompt = question
        else:
            has_options = True
            if _dataset_type(dataset) == "MCQ":
                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = ""
                for key, item in options.items():
                    options_prompt += f"{key}. {item}\n"
                hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
                if hint is not None:
                    prompt += f"Hint: {hint}\n"
                prompt += f"{question}\n"
                if len(options):
                    prompt += options_prompt
                else:
                    has_options = False

                if dataset is not None and "MMMU" in dataset and system_prompt:
                    prompt = system_prompt + "\n" + prompt
                    system_prompt = ""
            else:
                prompt = question

            if _dataset_type(dataset) in ["MCQ", "Y/N", "VQA"]:
                if _dataset_type(dataset) == "MCQ":
                    if has_options:
                        prompt = self.multi_choice_cot_prompt + prompt
                    else:
                        prompt = self.short_ans_cot_prompt + prompt
                elif _dataset_type(dataset) == "Y/N":
                    prompt = self.short_ans_cot_prompt + prompt
                elif dataset is not None and listinstr(["OCRBench"], dataset):
                    prompt = self.ocrbench_cot_prompt + prompt
                else:
                    prompt = self.short_ans_cot_prompt + prompt

        msgs = []
        if system_prompt:
            msgs.append(dict(type="text", value=system_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        msgs.append(dict(type="text", value=prompt))

        if dataset and dataset.startswith("MMMU_"):
            from ..dataset import MMMUDataset

            msgs = MMMUDataset.split_MMMU(msgs)

        return msgs

    def extract_answer(self, res, dataset=None):
        if dataset is None:
            return res
        if self._use_general_no_thinking_dataset(dataset):
            res = self._strip_leading_think_block(res)
            if _dataset_type(dataset) == "MCQ":
                mcq_answer = self._extract_mcq_answer_from_free_form(res)
                if mcq_answer:
                    return mcq_answer
            return res
        if self.use_cot(dataset):
            if _dataset_type(dataset) == "MCQ":
                pattern = r"Answer:\s*([A-Ia-i])(?![A-Za-z])"
                matches = re.findall(pattern, res, re.DOTALL)
                if matches:
                    return matches[-1].strip()
            elif _dataset_type(dataset) == "VQA" and not listinstr(["OCRBench", "MMVet"], dataset):
                pattern = r"Answer:\s*(.*)\s*$"
                match = re.search(pattern, res, re.DOTALL)
                if match:
                    return match.group(1)
            elif _dataset_type(dataset) == "Y/N":
                pattern = r"Answer:\s*(.*)\s*$"
                match = re.search(pattern, res, re.DOTALL)
                if match:
                    return match.group(1)
        return res

    def _strip_leading_think_block(self, text):
        text = str(text)
        stripped = re.sub(r"^\s*<think>\s*.*?\s*</think>\s*", "", text, count=1, flags=re.DOTALL)
        if stripped != text:
            return stripped.lstrip()
        return text.replace("<think>", "").replace("</think>", "").strip()

    def _extract_mcq_answer_from_free_form(self, text):
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

    def _message_to_content(self, message, dataset=None):
        content = []
        for item in message:
            if item["type"] == "text":
                content.append(item["value"])
            elif item["type"] == "image":
                image = Image.open(item["value"]).convert("RGB")
                if not self.use_upsize(dataset):
                    content.append(image)
                else:
                    img_width, img_height = image.width, image.height
                    if (img_width * img_height) >= (1344 * 1344):
                        content.append(image)
                    else:
                        ratio = math.sqrt((1344 * 1344) / (img_width * img_height))
                        max_img_width = int(img_width * ratio)
                        new_img_width = random.randint(img_width, max_img_width)
                        new_img_height = int(new_img_width / img_width * img_height)
                        content.append(image.resize((new_img_width, new_img_height)))
        return content

    def _message_to_vllm_content(self, message, dataset=None):
        content = []
        for item in message:
            if item["type"] == "text":
                content.append({"type": "text", "text": str(item["value"])})
            elif item["type"] == "image":
                image = Image.open(item["value"]).convert("RGB")
                if self.use_upsize(dataset):
                    img_width, img_height = image.width, image.height
                    if (img_width * img_height) < (1344 * 1344):
                        ratio = math.sqrt((1344 * 1344) / (img_width * img_height))
                        max_img_width = int(img_width * ratio)
                        new_img_width = random.randint(img_width, max_img_width)
                        new_img_height = int(new_img_width / img_width * img_height)
                        image = image.resize((new_img_width, new_img_height))
                content.append({"type": "image_pil", "image_pil": image})
        return content

    def _build_default_generate_kwargs(self, dataset=None):
        if self._should_disable_thinking(dataset):
            default_kwargs = dict(
                enable_thinking=False,
                max_new_tokens=self.max_new_tokens,
                sampling=False,
                num_beams=self.num_beams,
                repetition_penalty=1.2,
            )
        elif self.use_long_cot(dataset):
            default_kwargs = dict(
                enable_thinking=True,
                max_new_tokens=self.max_new_tokens,
                sampling=True,
                temperature=0.7,
                num_beams=1,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
            )
        elif self.use_cot(dataset):
            default_kwargs = dict(
                max_new_tokens=self.max_new_tokens,
                sampling=False,
                num_beams=self.num_beams,
                repetition_penalty=1.2,
            )
        else:
            default_kwargs = dict(
                max_new_tokens=self.max_new_tokens,
                sampling=False,
                num_beams=self.num_beams,
                repetition_penalty=1.2,
            )
        default_kwargs.update(self.kwargs)
        default_kwargs.update(self.generate_overrides)
        return default_kwargs

    def _build_vllm_sampling(self, dataset=None):
        from vllm import SamplingParams

        generate_kwargs = self._build_default_generate_kwargs(dataset=dataset)
        enable_thinking = generate_kwargs.pop("enable_thinking", None)
        max_new_tokens = int(generate_kwargs.pop("max_new_tokens", self.max_new_tokens))
        sampling = bool(generate_kwargs.pop("sampling", False))
        num_beams = int(generate_kwargs.pop("num_beams", 1))
        generate_kwargs.pop("no_repeat_ngram_size", None)
        if num_beams > 1 and not self._warned_vllm_beam_fallback:
            print(
                f"[MiniCPM_V_4_5] vLLM path approximates num_beams={num_beams} with greedy decoding.",
                flush=True,
            )
            self._warned_vllm_beam_fallback = True
        if sampling:
            temperature = float(generate_kwargs.pop("temperature", 0.7))
            top_p = float(generate_kwargs.pop("top_p", 1.0))
            top_k = int(generate_kwargs.pop("top_k", 0))
        else:
            temperature = 0.0
            top_p = 1.0
            top_k = 0
        repetition_penalty = float(generate_kwargs.pop("repetition_penalty", 1.0))
        presence_penalty = float(generate_kwargs.pop("presence_penalty", 0.0))
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
        )
        chat_template_kwargs = {}
        if enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = bool(enable_thinking)
        return sampling_params, chat_template_kwargs or None

    def _normalize_official_output(self, text):
        return str(text)

    def _extract_vllm_output(self, output):
        if getattr(output, "outputs", None):
            return output.outputs[0].text
        return ""

    def _restore_vllm_full_output(self, text, chat_template_kwargs=None):
        raw_text = str(text)
        if not chat_template_kwargs or not chat_template_kwargs.get("enable_thinking", False):
            return raw_text
        if raw_text.lstrip().startswith("<think>"):
            return raw_text
        return "<think>\n" + raw_text.lstrip("\n")

    def _build_structured_result(self, prediction, raw_output, full_output):
        return {
            "prediction": prediction,
            "description": "",
            "detailed_prediction": raw_output,
            "full_output": full_output,
        }

    def _generate_inner_vllm(self, message, dataset=None):
        return self._generate_batch_inner_vllm([message], dataset=dataset)[0]

    def _generate_batch_inner_vllm(self, messages, dataset=None):
        sampling_params, chat_template_kwargs = self._build_vllm_sampling(dataset=dataset)
        chat_template = None
        if chat_template_kwargs and chat_template_kwargs.get("enable_thinking") is False:
            chat_template = self._original_chat_template
        conversations = [
            [{"role": "user", "content": self._message_to_vllm_content(message, dataset=dataset)}]
            for message in messages
        ]
        outputs = self.llm.chat(
            conversations,
            sampling_params=sampling_params,
            use_tqdm=False,
            chat_template=chat_template,
            chat_template_content_format="string",
            chat_template_kwargs=chat_template_kwargs,
        )
        self._audit_text_filler_engine_outputs(messages, outputs, dataset=dataset)
        results = []
        for message, output in zip(messages, outputs):
            full_output = self._extract_vllm_output(output)
            raw_res = self._normalize_official_output(
                self._restore_vllm_full_output(
                    full_output,
                    chat_template_kwargs=chat_template_kwargs,
                )
            )
            final_res = self.extract_answer(raw_res, dataset)
            self._maybe_debug_log_io(dataset=dataset, message=message, raw_output=raw_res, final_output=final_res)
            results.append(self._build_structured_result(final_res, raw_res, full_output))
        return results

    def _clip_debug_text(self, text, limit):
        text = str(text)
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"

    def _serialize_message_for_debug(self, message):
        serialized = []
        total_items = len(message)
        for idx, item in enumerate(message[: self.debug_io_max_items]):
            if not isinstance(item, dict):
                serialized.append({"type": "unknown", "value": repr(item)})
                continue
            item_type = item.get("type", "unknown")
            value = item.get("value", "")
            if item_type == "text":
                serialized.append(
                    {
                        "type": "text",
                        "value": self._clip_debug_text(value, self.debug_io_max_text_chars),
                    }
                )
            else:
                serialized.append(
                    {
                        "type": item_type,
                        "value": str(value),
                    }
                )
        if total_items > self.debug_io_max_items:
            serialized.append(
                {
                    "type": "meta",
                    "value": f"... omitted {total_items - self.debug_io_max_items} items",
                }
            )
        return serialized

    def _maybe_debug_log_io(self, dataset, message, raw_output, final_output):
        self._debug_io_counter += 1
        if not self.debug_io_enabled:
            return
        if self._debug_io_counter % self.debug_io_every != 0:
            return

        payload = {
            "tag": "MINICPM_DEBUG_IO",
            "model": self.__class__.__name__,
            "call_index": self._debug_io_counter,
            "dataset": dataset,
            "input_message": self._serialize_message_for_debug(message),
            "raw_output": self._clip_debug_text(raw_output, self.debug_io_max_output_chars),
            "final_output": self._clip_debug_text(final_output, self.debug_io_max_output_chars),
        }
        try:
            print(f"[MINICPM_DEBUG_IO] {json.dumps(payload, ensure_ascii=False)}", flush=True)
        except Exception:
            print(
                f"[MINICPM_DEBUG_IO] model={self.__class__.__name__} call_index={self._debug_io_counter} dataset={dataset}",
                flush=True,
            )
        if self.debug_io_file:
            try:
                with open(self.debug_io_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception as err:
                print(
                    f"[MINICPM_DEBUG_IO] failed to write jsonl: {err}",
                    flush=True,
                )

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self._generate_inner_vllm(message, dataset=dataset)

        self.tokenizer.chat_template = self._select_hf_chat_template(dataset)

        default_kwargs = self._build_default_generate_kwargs(dataset=dataset)

        msgs = [{"role": "user", "content": self._message_to_content(message, dataset=dataset)}]
        self.processor.tokenizer = self.tokenizer
        res = self.model.chat(
            image=None,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_inp_length=8192,
            **default_kwargs,
        )

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]

        raw_res = self._normalize_official_output(res)
        final_res = self.extract_answer(raw_res, dataset)
        self._maybe_debug_log_io(dataset=dataset, message=message, raw_output=raw_res, final_output=final_res)
        return final_res

    def generate_batch_inner(self, messages, dataset=None):
        if self.use_vllm:
            return self._generate_batch_inner_vllm(messages, dataset=dataset)
        return [self.generate_inner(message, dataset=dataset) for message in messages]


class MiniCPM_V_4_5_Replay(MiniCPM_V_4_5):
    """Replay-enabled MiniCPM-V 4.5 with minimal message-level preprocessing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_cfg = read_replay_config_from_env()
        self.prompt_template_cfg = read_prompt_template_config_from_env()
        self.template_on_last_replay_text = os.environ.get(
            "REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        print(f"[MiniCPM_V_4_5_Replay] replay_cfg={self.replay_cfg}", flush=True)
        print(f"[MiniCPM_V_4_5_Replay] prompt_template_cfg={self.prompt_template_cfg}", flush=True)
        print(
            f"[MiniCPM_V_4_5_Replay] template_on_last_replay_text={self.template_on_last_replay_text}",
            flush=True,
        )

    def _apply_prompt_template_to_message(self, message, dataset=None):
        template_text = self.prompt_template_cfg.get("template", "{problem}")
        if template_text == "{problem}":
            return message

        target_idx = None
        for idx, item in enumerate(message):
            if not isinstance(item, dict):
                continue
            if item.get("type") != "text":
                continue
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

    def _normalize_message_order(self, message):
        vision_items = []
        text_items = []
        other_items = []
        for item in message:
            if not isinstance(item, dict):
                other_items.append(item)
                continue
            item_type = item.get("type")
            if item_type in {"image", "video"}:
                vision_items.append(item)
            elif item_type == "text":
                text_items.append(item)
            else:
                other_items.append(item)
        return vision_items + text_items + other_items

    def _apply_replay_pipeline(self, message, dataset=None):
        message = self._normalize_message_order(message)
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

    def _apply_text_filler_replay(self, message, dataset=None):
        """⟨V,T,F,T⟩: the ⟨V,T,T⟩ pipeline with a token-matched spaces block
        inserted between the two text halves.

        The vLLM chat path joins text parts of the user content with "\\n"
        (and prepends the image placeholder with the same separator), so the
        filler lands as its own part between the two question copies; the
        in-context token count of the filler is computed on that exact
        joined chunk.
        """
        base = self._apply_replay_pipeline_mode(message, dataset=dataset, mode="image_text_text")
        text_positions = [
            idx
            for idx, item in enumerate(base)
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if not text_positions or len(text_positions) % 2 != 0:
            raise RuntimeError(
                f"MiniCPM filler mode expected an even number of text items after "
                f"image_text_text expansion, got {len(text_positions)}"
            )
        half = len(text_positions) // 2
        first_text = "\n".join(str(base[idx]["value"]) for idx in text_positions[:half])
        second_text = "\n".join(str(base[idx]["value"]) for idx in text_positions[half:])
        image_items = [
            item
            for item in base
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        if not image_items:
            raise RuntimeError("MiniCPM filler mode requires at least one image item")
        visual_core, image_details = self._minicpm_visual_core_tokens(base)
        baseline_engine_prediction = self._predict_engine_prompt_len(base, dataset=dataset)

        def count_fn(filler_value):
            chunk_with = "\n" + first_text + "\n" + filler_value + "\n" + second_text
            chunk_without = "\n" + first_text + "\n" + second_text
            return count_filler_tokens(self.tokenizer, chunk_with, chunk_without)

        tag = f"{self.__class__.__name__} dataset={dataset}"
        spec = build_space_filler(count_fn, visual_core, tag=tag)
        filler_value = SPACE_FILLER_LITERAL * int(spec["chars"])
        filler_position = text_positions[half]
        filler_item = {
            "type": "text",
            "value": filler_value,
        }
        filler_message = []
        for idx, item in enumerate(base):
            if idx == filler_position:
                filler_message.append(filler_item)
            filler_message.append(item)
        filler_engine_prediction = self._predict_engine_prompt_len(
            filler_message, dataset=dataset
        )
        meta = {
            "dataset": str(dataset) if dataset is not None else None,
            "mode": REPLAY_IMAGE_TEXT_FILLER_TEXT,
            "image_count": len(image_items),
            "image_details": image_details,
            "visual_core_tokens": visual_core,
            "filler_chars": int(spec["chars"]),
            "filler_tokens_local": int(spec["tokens"]),
            "baseline_engine_len_predicted": baseline_engine_prediction,
            "filler_engine_len_predicted": filler_engine_prediction,
        }
        filler_item["text_filler_meta"] = dict(meta)
        self._text_filler_audit.log_construction({**meta, "phase_detail": "constructed"})
        return filler_message

    def generate_inner(self, message, dataset=None):
        replayed = self._apply_replay_pipeline(message, dataset=dataset)
        maybe_debug_print_replay(
            enabled=self.replay_cfg.get("debug", False),
            mode=canonicalize_replay_mode(self.replay_cfg.get("mode", "image_text")),
            before=message,
            after=replayed,
            tag=self.__class__.__name__,
        )
        return super().generate_inner(replayed, dataset=dataset)

    def generate_batch_inner(self, messages, dataset=None):
        replayed_messages = []
        replay_mode = canonicalize_replay_mode(self.replay_cfg.get("mode", "image_text"))
        for message in messages:
            replayed = self._apply_replay_pipeline(message, dataset=dataset)
            maybe_debug_print_replay(
                enabled=self.replay_cfg.get("debug", False),
                mode=replay_mode,
                before=message,
                after=replayed,
                tag=self.__class__.__name__,
            )
            replayed_messages.append(replayed)
        if self.use_vllm:
            return self._generate_batch_inner_vllm(replayed_messages, dataset=dataset)
        generate_one = super().generate_inner
        return [generate_one(message, dataset=dataset) for message in replayed_messages]
