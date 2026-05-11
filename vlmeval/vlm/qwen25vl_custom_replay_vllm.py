import json
import os
import random
import time

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from .base import BaseModel
from .qwen2_vl.prompt import Qwen2VLPromptMixin
from .replay_policy import (
    REPLAY_IMAGE_TEXT_TEXT,
    apply_replay,
    count_modalities,
    is_noop_replay_mode,
    maybe_debug_print_replay,
    read_replay_config_from_env,
)


class Qwen25VLCustomReplayvLLM(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path="", use_custom_prompt=True, **kwargs):
        super().__init__(use_custom_prompt=use_custom_prompt)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        from vllm.model_executor.models import ModelRegistry
        from .qwen2_5_vl_custom import Qwen2_5_VLCustomForConditionalGeneration

        ModelRegistry.register_model(
            "Qwen2_5_VLForConditionalGeneration",
            Qwen2_5_VLCustomForConditionalGeneration,
        )

        self.model_path = model_path
        self.replay_cfg = read_replay_config_from_env()

        tensor_parallel_size = torch.cuda.device_count()
        if tensor_parallel_size == 0:
            raise ValueError("VLLM requires at least one CUDA GPU.")

        # Keep old behavior for baseline, enlarge for sequence/image replay.
        default_limit = 1 if is_noop_replay_mode(self.replay_cfg["mode"]) or self.replay_cfg["mode"] == REPLAY_IMAGE_TEXT_TEXT else 8
        self.limit_mm_per_prompt = int(os.environ.get("REPLAY_LIMIT_MM_PER_PROMPT", str(default_limit)))

        print(
            f"Loading model from {self.model_path} using VLLM with tensor_parallel_size={tensor_parallel_size}",
            flush=True,
        )
        print(f"Replay config: {json.dumps(self.replay_cfg, ensure_ascii=False)}", flush=True)
        print(f"limit_mm_per_prompt.image={self.limit_mm_per_prompt}", flush=True)

        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.6,
            max_model_len=32768,
            limit_mm_per_prompt={"image": self.limit_mm_per_prompt, "video": 2},
            max_num_seqs=1,
            **kwargs,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self.sampling_params = SamplingParams(temperature=0.01, top_p=0.9, max_tokens=4096)

        self._speed_log_dir = os.environ.get("speed_log_dir", "").strip()
        self._speed_log_file = None
        if self._speed_log_dir:
            os.makedirs(self._speed_log_dir, exist_ok=True)
            self._speed_log_file = os.path.join(self._speed_log_dir, f"{self.__class__.__name__}.jsonl")
            print(f"[speed-log] enabled. Writing to {self._speed_log_file}", flush=True)
            self.sampling_params = SamplingParams(temperature=0.01, top_p=0.9, max_tokens=100)
            print(f"[speed-log] sampling_params changed to {self.sampling_params}", flush=True)

    def _extract_times(self, req_output, default_latency_s):
        result = {"ttft_s": None, "latency_s": default_latency_s, "decode_time_s": None}
        try:
            metrics = getattr(req_output, "metrics", None)
            if metrics is None:
                return result
            a = getattr(metrics, "arrival_time", None)
            f = getattr(metrics, "first_token_time", None)
            l = getattr(metrics, "last_token_time", None)
            fin = getattr(metrics, "finished_time", None)

            if a is not None and f is not None:
                result["ttft_s"] = max(0.0, float(f) - float(a))
            if a is not None and fin is not None:
                result["latency_s"] = max(1e-6, float(fin) - float(a))
            elif a is not None and l is not None:
                result["latency_s"] = max(1e-6, float(l) - float(a))
            if f is not None and l is not None:
                result["decode_time_s"] = max(1e-6, float(l) - float(f))
            elif f is not None and fin is not None:
                result["decode_time_s"] = max(1e-6, float(fin) - float(f))
        except Exception:
            pass
        return result

    def _write_speed_log(self, record):
        if not self._speed_log_file:
            return
        try:
            with open(self._speed_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[speed-log] write failed: {e}", flush=True)

    def _to_qwen_content(self, message):
        content = []
        for item in message:
            item_type = item["type"]
            if item_type == "text":
                content.append({"type": "text", "text": item["value"]})
            elif item_type == "image":
                content.append({"type": "image", "image": item["value"]})
            elif item_type == "video":
                content.append({"type": "video", "video": item["value"]})
        return content

    def _build_replayed_messages(self, message):
        base_content = self._to_qwen_content(message)
        replayed = apply_replay(
            base_content,
            mode=self.replay_cfg["mode"],
            repeat_times=self.replay_cfg["repeat_times"],
            image_copy_mode=self.replay_cfg["image_copy_mode"],
        )
        maybe_debug_print_replay(
            enabled=self.replay_cfg["debug"],
            mode=self.replay_cfg["mode"],
            before=base_content,
            after=replayed,
            tag=self.__class__.__name__,
        )
        return [{"role": "user", "content": replayed}], replayed

    def _build_single_llm_input(self, message):
        messages, replayed_content = self._build_replayed_messages(message)
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        text_prompt += "<think>\n\n"

        image_inputs, video_inputs = process_vision_info(messages)
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        if self.replay_cfg["debug"]:
            print(
                f"[REPLAY_DEBUG] prompt_chars={len(text_prompt)} replayed_counts={count_modalities(replayed_content)}",
                flush=True,
            )

        llm_inputs = {
            "prompt": text_prompt,
            "multi_modal_data": mm_data,
        }
        return llm_inputs, text_prompt

    def _split_prediction(self, output_text):
        if "</think>" in output_text:
            prediction = output_text.split("</think>", 1)[-1].strip()
            detailed_prediction = output_text
        else:
            prediction = output_text
            detailed_prediction = prediction
        return prediction, detailed_prediction

    def generate_inner(self, message, dataset=None):
        llm_inputs, text_prompt = self._build_single_llm_input(message)
        _t0 = time.perf_counter()
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        _t1 = time.perf_counter()
        total_latency_s = max(_t1 - _t0, 1e-6)

        output_text = outputs[0].outputs[0].text.strip()
        prediction, detailed_prediction = self._split_prediction(output_text)

        try:
            out_seq = outputs[0].outputs[0]
            out_token_ids = getattr(out_seq, "token_ids", []) or []
            output_len = max(len(out_token_ids) - 1, 0)

            times = self._extract_times(outputs[0], total_latency_s)
            ttft_s = times["ttft_s"]
            latency_s = times["latency_s"]
            decode_time_s = times["decode_time_s"]
            if decode_time_s is None:
                decode_time_s = (latency_s - ttft_s) if (ttft_s is not None and latency_s >= ttft_s) else latency_s
            decode_time_s = max(decode_time_s, 1e-6)
            avg_decode_tps = (output_len / decode_time_s) if output_len > 0 else 0.0

            if self._speed_log_dir:
                self._write_speed_log(
                    {
                        "ts": time.time(),
                        "type": "single",
                        "model": self.model_path,
                        "dataset": str(dataset) if dataset is not None else None,
                        "prompt_chars": len(text_prompt),
                        "output_tokens": output_len,
                        "ttft_ms": (ttft_s * 1000.0) if ttft_s is not None else None,
                        "latency_ms": latency_s * 1000.0,
                        "avg_decode_tps": avg_decode_tps,
                        "replay_mode": self.replay_cfg["mode"],
                    }
                )
        except Exception as e:
            print(f"[speed-log] single record failed: {e}", flush=True)

        return {
            "prediction": prediction,
            "detailed_prediction": detailed_prediction,
        }

    def generate_batch_inner(self, messages, dataset=None):
        batch_llm_inputs = []
        text_prompts = []
        for message in messages:
            llm_inputs, text_prompt = self._build_single_llm_input(message)
            batch_llm_inputs.append(llm_inputs)
            text_prompts.append(text_prompt)

        _t0 = time.perf_counter()
        outputs = self.llm.generate(batch_llm_inputs, sampling_params=self.sampling_params)
        _t1 = time.perf_counter()
        total_latency_s = max(_t1 - _t0, 1e-6)

        results_list = []
        for idx, output in enumerate(outputs):
            output_text = output.outputs[0].text.strip()
            prediction, detailed_prediction = self._split_prediction(output_text)
            results_list.append(
                {
                    "prediction": prediction,
                    "detailed_prediction": detailed_prediction,
                }
            )

            try:
                out_seq = output.outputs[0]
                out_token_ids = getattr(out_seq, "token_ids", []) or []
                output_len = max(len(out_token_ids) - 1, 0)
                times = self._extract_times(output, total_latency_s)
                ttft_s = times["ttft_s"]
                latency_s = times["latency_s"]
                decode_time_s = times["decode_time_s"]
                if decode_time_s is None:
                    decode_time_s = (latency_s - ttft_s) if (ttft_s is not None and latency_s >= ttft_s) else latency_s
                decode_time_s = max(decode_time_s, 1e-6)
                avg_decode_tps = (output_len / decode_time_s) if output_len > 0 else 0.0

                if self._speed_log_dir:
                    self._write_speed_log(
                        {
                            "ts": time.time(),
                            "type": "batch",
                            "index": idx,
                            "model": self.model_path,
                            "dataset": str(dataset) if dataset is not None else None,
                            "prompt_chars": len(text_prompts[idx]) if idx < len(text_prompts) else None,
                            "output_tokens": output_len,
                            "ttft_ms": (ttft_s * 1000.0) if ttft_s is not None else None,
                            "latency_ms": latency_s * 1000.0,
                            "avg_decode_tps": avg_decode_tps,
                            "replay_mode": self.replay_cfg["mode"],
                        }
                    )
            except Exception as e:
                print(f"[speed-log] batch record failed for idx={idx}: {e}", flush=True)

        return results_list
