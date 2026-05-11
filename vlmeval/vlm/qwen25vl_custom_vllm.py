import math
import torch
import torch.distributed as dist
import random
import numpy as np
from PIL import Image
import os
import time
import json

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE

from .qwen2_vl.prompt import Qwen2VLPromptMixin

class Qwen25VLCustomvLLM(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='', use_custom_prompt=True, **kwargs):
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

        tensor_parallel_size = torch.cuda.device_count()
        if tensor_parallel_size == 0:
            raise ValueError("VLLM requires at least one CUDA GPU.")

        print(f'Loading model from {self.model_path} using VLLM with tensor_parallel_size={tensor_parallel_size}')

        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.6,
            max_model_len=32768,
            limit_mm_per_prompt={"image": 1, "video": 0},
            max_num_seqs=1,
            **kwargs
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=4096)
        self.sampling_params = SamplingParams(temperature=0.01, top_p=0.9, max_tokens=4096)

        # Optional speed logging
        self._speed_log_dir = os.environ.get("speed_log_dir", "").strip()
        self._speed_log_file = None
        if self._speed_log_dir:
            os.makedirs(self._speed_log_dir, exist_ok=True)
            self._speed_log_file = os.path.join(self._speed_log_dir, f"{self.__class__.__name__}.jsonl")
            print(f"[speed-log] enabled. Writing to {self._speed_log_file}", flush=True)

            # change max_tokens to 100 for speed test
            self.sampling_params = SamplingParams(temperature=0.01, top_p=0.9, max_tokens=100)
            print(f"[speed-log] sampling_params changed to {self.sampling_params}", flush=True)

    def _extract_times(self, req_output, default_latency_s):
        # 提取 TTFT、总耗时、解码时长（秒）
        result = {"ttft_s": None, "latency_s": default_latency_s, "decode_time_s": None}
        try:
            metrics = getattr(req_output, "metrics", None)
            if metrics is None:
                return result

            a = getattr(metrics, "arrival_time", None)
            f = getattr(metrics, "first_token_time", None)
            l = getattr(metrics, "last_token_time", None)
            fin = getattr(metrics, "finished_time", None)
            fs = getattr(metrics, "first_scheduled_time", None)

            # TTFT = 首 token - 到达
            if a is not None and f is not None:
                result["ttft_s"] = max(0.0, float(f) - float(a))

            # 总耗时优先 finished - arrival，其次 last - arrival
            if a is not None and fin is not None:
                result["latency_s"] = max(1e-6, float(fin) - float(a))
            elif a is not None and l is not None:
                result["latency_s"] = max(1e-6, float(l) - float(a))

            # 解码时长 = last - first，或 finished - first
            if f is not None and l is not None:
                result["decode_time_s"] = max(1e-6, float(l) - float(f))
            elif f is not None and fin is not None:
                result["decode_time_s"] = max(1e-6, float(fin) - float(f))

        except Exception:
            pass
        return result

    def _write_speed_log(self, record: dict):
        if not self._speed_log_file:
            return
        try:
            with open(self._speed_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            # 不影响主流程
            print(f"[speed-log] write failed: {e}", flush=True)

    def generate_inner(self, message, dataset=None):
        print(f"Generating response for message: {message}", flush=True)

        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text_prompt += "<think>\n\n"

        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": text_prompt,
            "multi_modal_data": mm_data,
        }

        _t0 = time.perf_counter()
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        _t1 = time.perf_counter()
        total_latency_s = max(_t1 - _t0, 1e-6)

        output_text = outputs[0].outputs[0].text.strip()

        if "</think>" in output_text:
            prediction = output_text.split("</think>", 1)[-1].strip()
            detailed_prediction = output_text
        else:
            prediction = output_text
            detailed_prediction = prediction

        # speed log (single)
        try:
            out_seq = outputs[0].outputs[0]
            out_token_ids = getattr(out_seq, "token_ids", []) or []
            output_len = len(out_token_ids)
            output_len = output_len - 1 # decode time 不包含第一个 token

            times = self._extract_times(outputs[0], total_latency_s)
            ttft_s = times["ttft_s"]
            latency_s = times["latency_s"]
            decode_time_s = times["decode_time_s"]

            # 兜底回退
            if decode_time_s is None:
                decode_time_s = (latency_s - ttft_s) if (ttft_s is not None and latency_s >= ttft_s) else latency_s
            decode_time_s = max(decode_time_s, 1e-6)

            avg_decode_tps = (output_len / decode_time_s) if output_len > 0 else 0.0

            if self._speed_log_dir:
                self._write_speed_log({
                    "ts": time.time(),
                    "type": "single",
                    "model": self.model_path,
                    "dataset": str(dataset) if dataset is not None else None,
                    "image_path": image_path,
                    "prompt_chars": len(text_prompt),
                    "output_tokens": output_len,
                    "ttft_ms": (ttft_s * 1000.0) if ttft_s is not None else None,
                    "latency_ms": latency_s * 1000.0,
                    "avg_decode_tps": avg_decode_tps,
                })
        except Exception as e:
            print(f"[speed-log] single record failed: {e}", flush=True)

        res = {
            'prediction': prediction,
            'detailed_prediction': detailed_prediction
        }

        return res

    def generate_batch_inner(self, messages, dataset=None):
        print(f"Generating batch with {len(messages)} messages.", flush=True)

        prompts, image_paths = self.messages_to_promptimgs(messages, dataset=dataset)

        batch_llm_inputs = []
        text_prompts = []
        for prompt, image_path in zip(prompts, image_paths):
            current_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text_prompt = self.processor.apply_chat_template(
                current_messages, tokenize=False, add_generation_prompt=True
            )
            text_prompt += "<think>\n\n"
            text_prompts.append(text_prompt)

            image_inputs, video_inputs = process_vision_info(current_messages)

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            llm_inputs = {
                "prompt": text_prompt,
                "multi_modal_data": mm_data,
            }
            batch_llm_inputs.append(llm_inputs)

        _t0 = time.perf_counter()
        outputs = self.llm.generate(batch_llm_inputs, sampling_params=self.sampling_params)
        _t1 = time.perf_counter()
        total_latency_s = max(_t1 - _t0, 1e-6)

        results_list = []
        for idx, output in enumerate(outputs):
            output_text = output.outputs[0].text.strip()

            if "</think>" in output_text:
                prediction = output_text.split("</think>", 1)[-1].strip()
                detailed_prediction = output_text
            else:
                prediction = output_text
                detailed_prediction = prediction

            results_list.append({
                'prediction': prediction,
                'detailed_prediction': detailed_prediction,
            })

            # speed log (batch per-sample)
            try:
                out_seq = output.outputs[0]
                out_token_ids = getattr(out_seq, "token_ids", []) or []
                output_len = len(out_token_ids)
                output_len = output_len - 1 # decode time 不包含第一个 token

                times = self._extract_times(output, total_latency_s)
                ttft_s = times["ttft_s"]
                latency_s = times["latency_s"]
                decode_time_s = times["decode_time_s"]

                if decode_time_s is None:
                    decode_time_s = (latency_s - ttft_s) if (ttft_s is not None and latency_s >= ttft_s) else latency_s
                decode_time_s = max(decode_time_s, 1e-6)

                avg_decode_tps = (output_len / decode_time_s) if output_len > 0 else 0.0

                if self._speed_log_dir:
                    self._write_speed_log({
                        "ts": time.time(),
                        "type": "batch",
                        "index": idx,
                        "model": self.model_path,
                        "dataset": str(dataset) if dataset is not None else None,
                        "image_path": image_paths[idx] if idx < len(image_paths) else None,
                        "prompt_chars": len(text_prompts[idx]) if idx < len(text_prompts) else None,
                        "output_tokens": output_len,
                        "ttft_ms": (ttft_s * 1000.0) if ttft_s is not None else None,
                        "latency_ms": latency_s * 1000.0,
                        "avg_decode_tps": avg_decode_tps,
                    })
            except Exception as e:
                print(f"[speed-log] batch record failed for idx={idx}: {e}", flush=True)

        return results_list