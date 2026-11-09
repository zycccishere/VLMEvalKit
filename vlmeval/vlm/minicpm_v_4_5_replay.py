import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any

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
from .replay_image_transform import (
    apply_image_transform_to_content,
    canonicalize_image_transform,
    image_rgb_sha256,
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


def _env_optional_int(name):
    """Optional int env: None when unset/empty; invalid values warn and
    return None (the knob is silently disabled rather than crashing a run
    that never asked for determinism)."""
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return None
    if not raw.lstrip("+-").isdigit():
        print(f"[MiniCPM_V_4_5] ignoring invalid int env {name}={raw!r}", flush=True)
        return None
    return int(raw)


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
                # llm.chat below uses chat_template_content_format="string".
                # With vLLM's default (False) its chat parser moves every image
                # placeholder to the front of the joined text, so replay orders
                # such as <V,T,V,T> reach the engine as <V,V,T,T>. Interleaving
                # keeps each placeholder at its content position, matching the
                # checkpoint's own HF chat(); image-first content renders
                # byte-identically either way.
                interleave_mm_strings=True,
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
            with Image.open(self._strip_file_scheme(item["value"])) as handle:
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

    def _expand_image_placeholder_text(self, size):
        image_processor = getattr(self.processor, "image_processor", None)
        return image_processor.get_slice_image_placeholder(size)

    def _render_message_prompt_str(self, message, dataset=None):
        """Chat-template string for ``message`` with the literal image
        placeholder pattern (pre-replacement), mirroring llm.chat."""
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
        return self._render_vllm_chat_string(content, dataset=dataset)

    def _predict_engine_prompt_len(self, message, dataset=None):
        """Local replica of the engine's prompt length for ``message``.

        Mirrors vLLM's MiniCPMV prompt-update path: the chat string P (with
        the literal ``(<image>./</image>)`` pattern once per image) is
        tokenized as a whole, then each pattern's token span is replaced by
        the expanded placeholder's tokens. No bos is prepended (this
        tokenizer has ``add_bos_token=False`` and vLLM adds nothing else).
        A constant residual scaffold offset between this replica and the
        engine is expected (e.g. from BPE merges at the pattern boundary)
        and is verified constant per batch by the engine audit.
        """
        prompt_str = self._render_message_prompt_str(message, dataset=dataset)
        image_items = [
            item
            for item in message
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        length = len(self.tokenizer.encode(prompt_str))
        if image_items:
            pattern_len = tokenize_length(
                self.tokenizer, self._minicpm_image_placeholder_pattern()
            )
            for item in image_items:
                with Image.open(self._strip_file_scheme(item["value"])) as handle:
                    width, height = handle.size
                placeholder = self._expand_image_placeholder_text((width, height))
                length += tokenize_length(self.tokenizer, placeholder) - pattern_len
        return length

    def _minicpm_unk_token_id(self) -> int:
        image_processor = getattr(self.processor, "image_processor", None)
        literal = getattr(image_processor, "unk_token", None) or getattr(
            self.tokenizer, "unk_token", "<unk>"
        )
        token_id = int(self.tokenizer.convert_tokens_to_ids(str(literal)))
        if token_id is None or token_id < 0:
            raise RuntimeError(f"cannot resolve MiniCPM image feature token {literal!r}")
        return token_id

    def _count_pure_space_tokens(self, ids):
        """Engine-side count of tokens whose decoded text is entirely space
        characters (U+0020 only); see the Gemma3 wrapper for the generalized
        literal form."""
        return self._count_pure_literal_tokens(ids, SPACE_FILLER_LITERAL)

    def _count_pure_literal_tokens(self, ids, literal):
        """Engine-side count of tokens whose decoded text consists solely of
        ``literal`` (generalizes the pure-space audit to the dots filler).

        The filler block is a run of a single literal character, so its
        tokens decode to pure-literal strings; normal question text never
        produces them (spaces merge into word tokens that decode with the
        word attached, and newline/tab tokens fail the single-literal set
        check).
        """
        count = 0
        for token_id in ids:
            text = self.tokenizer.decode([token_id])
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
        unk_id = self._minicpm_unk_token_id()
        records = []
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
            filler_literal = str(meta.get("filler_literal", SPACE_FILLER_LITERAL))
            records.append(
                {
                    **meta,
                    "engine_prompt_len": engine_len,
                    "engine_prompt_len_predicted": filler_prediction,
                    "scaffold_offset": engine_len - filler_prediction,
                    "engine_image_core_count": unk_count,
                    "filler_tokens_engine_spaces": self._count_pure_space_tokens(ids)
                    - int(meta.get("baseline_pure_space_tokens", 0)),
                    "filler_tokens_engine_literal": self._count_pure_literal_tokens(
                        ids, filler_literal
                    )
                    - int(
                        meta.get(
                            "baseline_pure_literal_tokens",
                            meta.get("baseline_pure_space_tokens", 0),
                        )
                    ),
                }
            )
        offsets = {int(r["scaffold_offset"]) for r in records}
        if len(offsets) > 1:
            self._dump_filler_mismatch_debug(records[0], messages, outputs, dataset)
            raise RuntimeError(
                f"[text-filler] ENGINE AUDIT MISMATCH ({self.__class__.__name__}, "
                f"dataset={dataset}): scaffold offset not constant within batch: "
                f"{sorted(offsets)}; records={records[:3]}"
            )
        offset = offsets.pop()
        for record in records:
            visual_core = int(record["visual_core_tokens"])
            literal_tokens = int(record["filler_tokens_engine_literal"])
            reasons = []
            if abs(offset) > 8:
                reasons.append(f"scaffold offset {offset} exceeds tolerance 8")
            if record["engine_image_core_count"] != visual_core:
                reasons.append(
                    f"engine image core count {record['engine_image_core_count']} != {visual_core}"
                )
            # Pure-literal tokens lose at most the two boundary merges of the
            # filler run against the surrounding "\n" separators; anything
            # beyond that is a real tokenization mismatch.
            if not (-2 <= literal_tokens - visual_core <= 0):
                reasons.append(
                    f"engine pure-literal filler tokens {literal_tokens} deviate from {visual_core} "
                    f"by {literal_tokens - visual_core} (tolerance [-2, 0])"
                )
            record["engine_audit_pass"] = not reasons
            record["engine_audit_reason"] = "; ".join(reasons) or "ok"
            if reasons:
                self._dump_filler_mismatch_debug(record, messages, outputs, dataset)
            self._text_filler_audit.log_engine(record)

    def _dump_filler_mismatch_debug(self, record, messages, outputs, dataset=None):
        debug_dir = os.environ.get("REPLAY_TEXT_FILLER_DEBUG_DIR", "/tmp").strip() or "/tmp"
        try:
            os.makedirs(debug_dir, exist_ok=True)
            tag = f"mismatch_{self.__class__.__name__}_{dataset}".replace("/", "_")
            with open(os.path.join(debug_dir, f"{tag}_record.json"), "w", encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=False, default=str)
        except Exception as dump_err:
            print(f"[text-filler] mismatch debug dump failed: {dump_err}", flush=True)

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
        if _env_truthy("MINICPM45_UPSIZE", False):
            # 2026-11-08 upsize probe (env-gated April-era policy restore):
            # the pre-disable form (357d93d6) applied upsize to long-CoT
            # datasets plus AI2D/OCRBench/ChartQA/TextVQA. The resize itself
            # lives in _message_to_content / _message_to_vllm_content:
            # images with w*h < 1344^2 are scaled to that area with a random
            # width (random.randint); the module seeds random at import, so
            # runs are reproducible. Unset -> release behavior (False),
            # byte-identical.
            if dataset is None:
                return False
            if self.use_long_cot(dataset):
                return True
            return listinstr(["AI2D", "OCRBench", "ChartQA", "TextVQA"], dataset)
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
        images = []
        for item in message:
            if item["type"] == "text":
                content.append(item["value"])
            elif item["type"] == "image":
                image = Image.open(self._strip_file_scheme(item["value"])).convert("RGB")
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
                if content and isinstance(content[-1], Image.Image):
                    images.append(content[-1])
        self._maybe_dump_minicpm_payload(
            phase="minicpm_hf_payload",
            dataset=dataset,
            message=message,
            images=images,
        )
        return content

    def _message_to_vllm_content(self, message, dataset=None):
        content = []
        images = []
        for item in message:
            if item["type"] == "text":
                content.append({"type": "text", "text": str(item["value"])})
            elif item["type"] == "image":
                image = Image.open(self._strip_file_scheme(item["value"])).convert("RGB")
                if self.use_upsize(dataset):
                    img_width, img_height = image.width, image.height
                    if (img_width * img_height) < (1344 * 1344):
                        ratio = math.sqrt((1344 * 1344) / (img_width * img_height))
                        max_img_width = int(img_width * ratio)
                        new_img_width = random.randint(img_width, max_img_width)
                        new_img_height = int(new_img_width / img_width * img_height)
                        image = image.resize((new_img_width, new_img_height))
                images.append(image)
                content.append({"type": "image_pil", "image_pil": image})
        self._maybe_dump_minicpm_payload(
            phase="minicpm_vllm_payload",
            dataset=dataset,
            message=message,
            images=images,
        )
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
        # Optional per-request sampling determinism for multi-seed repeats of
        # sampled long-CoT conditions (engine determinism probes C3/C4).
        # Unset (default) leaves SamplingParams.seed at vLLM's default.
        sampling_seed = _env_optional_int("VLMEVAL_SAMPLING_SEED")
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            **({"seed": sampling_seed} if sampling_seed is not None else {}),
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

    @staticmethod
    def _strip_file_scheme(image_ref):
        raw = str(image_ref or "").strip()
        if raw.startswith("file://"):
            return raw[len("file://") :]
        return raw

    def _maybe_dump_minicpm_payload(self, *, phase, dataset, message, images):
        dump = getattr(self, "_write_replay_dump", None)
        if not callable(dump):
            return
        active_ids = getattr(self, "_minicpm_trace_active_message_ids", set())
        message_id = id(message)
        if message_id not in active_ids:
            return
        transform_record = getattr(self, "_minicpm_transform_records_by_message_id", {}).get(
            message_id,
            getattr(self, "_last_image_transform_records", None),
        )
        dump(
            {
                "phase": phase,
                "dataset": str(dataset) if dataset is not None else None,
                "payload_image_count": len(images),
                "payload_image_sizes": [list(image.size) for image in images],
                "payload_image_rgb_sha256": [
                    image_rgb_sha256(image) for image in images
                ],
                "message_replayed": self._serialize_message_for_debug(message),
                "transform_records": transform_record if isinstance(transform_record, list) else None,
            },
            detail="summary",
            force=True,
        )
        active_ids.discard(message_id)
        getattr(self, "_minicpm_transform_records_by_message_id", {}).pop(message_id, None)

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
        self.image_transform_name = canonicalize_image_transform(os.environ.get("REPLAY_IMAGE_TRANSFORM", "baseline"))
        self.image_transform_cache_dir = os.environ.get("REPLAY_IMAGE_TRANSFORM_CACHE_DIR", "").strip()
        self.image_transform_effective_cache_dir = self.image_transform_cache_dir or os.path.join(
            os.getcwd(),
            ".replay_transform_cache",
        )
        # Comma-separated image positions (1-based, counted among image items)
        # to apply the transform to. The single-position form "2" is the
        # legacy paper path (blank the second image of image_text_image_text);
        # "2,3" / "2,3,4" cover the multi-image-copy SLP tiers.
        raw_positions = str(os.environ.get("REPLAY_IMAGE_TRANSFORM_TARGET_POSITION", "2") or "2")
        positions = []
        for part in raw_positions.split(","):
            part = part.strip()
            if part.isdigit():
                positions.append(max(1, int(part)))
        self.image_transform_target_positions = tuple(sorted(set(positions))) or (2,)
        self._last_image_transform_records: list[dict[str, Any]] | None = None
        self._minicpm_transform_records_by_message_id: dict[int, list[dict[str, Any]]] = {}
        self._minicpm_trace_active_message_ids: set[int] = set()
        trace_level = os.environ.get("REPLAY_TRACE_LEVEL", os.environ.get("REPLAY_STAGE_DEBUG", "off")).strip().lower()
        if trace_level in {"1", "true", "yes", "on"}:
            trace_level = "summary"
        if trace_level not in {"off", "summary", "full"}:
            trace_level = "off"
        self._minicpm_trace_level = trace_level
        self._minicpm_trace_max_samples = max(0, _env_int("REPLAY_TRACE_SAMPLES", 3))
        self._minicpm_trace_seen_samples = 0
        self._minicpm_trace_active = False
        self._minicpm_dump_dir = os.environ.get("REPLAY_TRACE_DIR", os.environ.get("REPLAY_DUMP_DIR", "")).strip()
        self._minicpm_dump_file = None
        self._minicpm_dump_max_chars = _env_int("REPLAY_TRACE_MAX_CHARS", _env_int("REPLAY_DUMP_MAX_CHARS", 0))
        if self._minicpm_dump_dir:
            os.makedirs(self._minicpm_dump_dir, exist_ok=True)
            pid = os.getpid()
            self._minicpm_dump_file = os.path.join(
                self._minicpm_dump_dir,
                f"{self.__class__.__name__}.pid{pid}.jsonl",
            )
            print(f"[minicpm-replay-dump] enabled. Writing to {self._minicpm_dump_file}", flush=True)
        print(
            f"[MiniCPM_V_4_5_Replay] image_transform={self.image_transform_name} "
            f"target_positions={self.image_transform_target_positions} "
            f"cache_dir={self.image_transform_effective_cache_dir}",
            flush=True,
        )

    def _begin_trace_sample(self):
        if self._minicpm_trace_level in {"summary", "full"} and (
            self._minicpm_trace_seen_samples < self._minicpm_trace_max_samples
        ):
            self._minicpm_trace_seen_samples += 1
            self._minicpm_trace_active = True
        else:
            self._minicpm_trace_active = False
        return self._minicpm_trace_active

    def _trace_allows(self, detail="summary", *, force=False):
        if not self._minicpm_dump_file:
            return False
        if not force and not self._minicpm_trace_active:
            return False
        if self._minicpm_trace_level == "full":
            return True
        return detail != "full"

    def _clip_trace_text(self, value):
        text = str(value)
        if self._minicpm_dump_max_chars > 0 and len(text) > self._minicpm_dump_max_chars:
            return text[: self._minicpm_dump_max_chars] + f"\n...[TRUNCATED {len(text) - self._minicpm_dump_max_chars} chars]"
        return text

    def _write_replay_dump(self, record, detail="summary", *, force=False):
        if not self._trace_allows(detail, force=force):
            return
        serializable = json.loads(json.dumps(record, default=str, ensure_ascii=False))
        serializable.setdefault("pid", os.getpid())
        serializable.setdefault("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        serializable.setdefault("rank", os.environ.get("RANK", os.environ.get("LOCAL_RANK", "")))
        for key in ("prompt", "raw_output", "full_output"):
            if key in serializable:
                serializable[key] = self._clip_trace_text(serializable[key])
        try:
            with open(self._minicpm_dump_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(serializable, ensure_ascii=False) + "\n")
        except Exception as err:
            print(f"[minicpm-replay-dump] write failed: {err}", flush=True)

    def _extract_replay_meta(self, inputs):
        for item in inputs:
            if isinstance(item, dict) and isinstance(item.get("replay_meta"), dict):
                return dict(item["replay_meta"])
        return {}

    def _apply_image_transform_pipeline(self, message, *, inputs, dataset=None, trace_active=False):
        if self.image_transform_name == "baseline":
            self._last_image_transform_records = [{"transform": "baseline", "applied": False}]
            if trace_active:
                self._minicpm_trace_active_message_ids.add(id(message))
            return message
        replay_meta = self._extract_replay_meta(inputs)
        transformed = message
        records: list[dict[str, Any]] = []
        for position in self.image_transform_target_positions:
            transformed, transform_record = apply_image_transform_to_content(
                transformed,
                transform_name=self.image_transform_name,
                sample_meta=replay_meta,
                cache_dir=self.image_transform_effective_cache_dir,
                dataset_name=str(dataset) if dataset is not None else "unknown_dataset",
                image_position=position,
                model_family="minicpm45",
            )
            records.append(transform_record)
        self._last_image_transform_records = records
        if trace_active:
            self._minicpm_transform_records_by_message_id[id(transformed)] = records
            self._minicpm_trace_active_message_ids.add(id(transformed))
            self._write_replay_dump(
                {
                    "phase": "image_transform",
                    "dataset": str(dataset) if dataset is not None else None,
                    "image_transform": self.image_transform_name,
                    "target_positions": list(self.image_transform_target_positions),
                    "replay_meta": replay_meta,
                    "records": records,
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
        """⟨V,T,F,T⟩: the ⟨V,T,T⟩ pipeline with a token-matched filler block
        inserted between the two text halves.

        The vLLM chat path joins text parts of the user content with "\\n"
        (and prepends the image placeholder with the same separator), so the
        filler lands as its own part between the two question copies; the
        in-context token count of the filler is computed on that exact
        joined chunk. The carrier literal is selected by
        ``REPLAY_TEXT_FILLER_TYPE`` (spaces/dots); an unknown value is a
        hard error in the shared module.
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
        filler_type = read_text_filler_type()
        filler_literal = filler_literal_for(filler_type)
        baseline_engine_prediction = self._predict_engine_prompt_len(base, dataset=dataset)
        baseline_prompt_str = self._render_message_prompt_str(base, dataset=dataset)
        baseline_ids = self.tokenizer(baseline_prompt_str, add_special_tokens=False).input_ids
        baseline_pure_space = self._count_pure_space_tokens(baseline_ids)
        baseline_pure_literal = self._count_pure_literal_tokens(baseline_ids, filler_literal)

        def count_fn(filler_value):
            chunk_with = "\n" + first_text + "\n" + filler_value + "\n" + second_text
            chunk_without = "\n" + first_text + "\n" + second_text
            return count_filler_tokens(self.tokenizer, chunk_with, chunk_without)

        tag = f"{self.__class__.__name__} dataset={dataset}"
        spec = build_text_filler(count_fn, visual_core, literal=filler_literal, tag=tag)
        filler_value = filler_literal * int(spec["chars"])
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
            "filler_type": filler_type,
            "filler_literal": filler_literal,
            "image_count": len(image_items),
            "image_details": image_details,
            "visual_core_tokens": visual_core,
            "filler_chars": int(spec["chars"]),
            "filler_tokens_local": int(spec["tokens"]),
            "baseline_engine_len_predicted": baseline_engine_prediction,
            "baseline_pure_space_tokens": baseline_pure_space,
            "baseline_pure_literal_tokens": baseline_pure_literal,
            "filler_engine_len_predicted": filler_engine_prediction,
        }
        filler_item["text_filler_meta"] = dict(meta)
        self._text_filler_audit.log_construction({**meta, "phase_detail": "constructed"})
        return filler_message

    def generate_inner(self, message, dataset=None):
        trace_active = self._begin_trace_sample()
        replayed = self._apply_replay_pipeline(message, dataset=dataset)
        replayed = self._apply_image_transform_pipeline(
            replayed,
            inputs=message,
            dataset=dataset,
            trace_active=trace_active,
        )
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
            trace_active = self._begin_trace_sample()
            replayed = self._apply_replay_pipeline(message, dataset=dataset)
            replayed = self._apply_image_transform_pipeline(
                replayed,
                inputs=message,
                dataset=dataset,
                trace_active=trace_active,
            )
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
