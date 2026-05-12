import math
import torch
import torch.distributed as dist
import random
import numpy as np
from PIL import Image
import time
import json

from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE

from .qwen2_vl.prompt import Qwen2VLPromptMixin

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
from qwen_vl_utils import process_vision_info

import os
Temperature = float(os.getenv('TEMPERATURE', 0.01))
HIGH_TEMP = os.getenv('HIGH_TEMP', 'False')
CUSTOM_PREFIX = os.environ.get("CUSTOM_PREFIX", "<think>\n\n</think>\n\n")

class Qwen25VLCustomPrefixCustom(Qwen2VLPromptMixin, BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='', use_custom_prompt=True, **kwargs):
        super().__init__(use_custom_prompt=use_custom_prompt)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        self.model_path = model_path

        tensor_parallel_size = torch.cuda.device_count()
        if tensor_parallel_size == 0:
            raise ValueError("Requires at least one CUDA GPU.")

        print(f'Loading models from {self.model_path}')

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.model = self.model.eval()

        self.kwargs = kwargs

        print(f"Using Temperature: {Temperature}")
        print(f"Using HIGH_TEMP: {HIGH_TEMP}")

        if HIGH_TEMP.lower() == 'false':
            self.sampling_params = dict(do_sample=True, temperature=0.01, top_p=0.001, top_k=1, max_new_tokens=4096, repetition_penalty=1.0) # original
        else:
            self.sampling_params = dict(do_sample=True, temperature=0.7, top_p=0.9, top_k=50, max_new_tokens=4096, repetition_penalty=1.0)

        # 测速日志相关初始化
        self._speed_log_dir = os.environ.get("speed_log_dir", "").strip()
        self._speed_log_file = None
        if self._speed_log_dir:
            os.makedirs(self._speed_log_dir, exist_ok=True)
            self._speed_log_file = os.path.join(self._speed_log_dir, f"{self.__class__.__name__}.jsonl")
            print(f"[speed-log] enabled. Writing to {self._speed_log_file}", flush=True)

            # change max_new_tokens to 100 for speed test
            self.sampling_params['max_new_tokens'] = 100
            print(f"[speed-log] max_new_tokens changed to {self.sampling_params['max_new_tokens']}", flush=True)

    def _write_speed_log(self, record: dict):
        if not self._speed_log_file:
            return
        try:
            with open(self._speed_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            # 不影响主流程
            print(f"[speed-log] write failed: {e}", flush=True)

    # def _resize_image_if_needed(self, image: Image.Image) -> Image.Image:
    #     w, h = image.size
    #     if w >= self.min_image_size and h >= self.min_image_size:
    #         return image

    #     min_dim = min(w, h)
    #     scale_factor = self.min_image_size / min_dim

    #     new_w = int(w * scale_factor) + 1
    #     new_h = int(h * scale_factor) + 1

    #     resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    #     print(f"Warning: Resized image from ({w}, {h}) to ({new_w}, {new_h}) to meet model requirements.")

    #     return resized_image

    def generate_inner(self, message, dataset=None):
        print(f"Generating response for message: {message}", flush=True)

        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text = text + CUSTOM_PREFIX

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # 如果启用测速日志，先运行一次 max_new_tokens=1 来测试 TTFT
        ttft_s = None
        if self._speed_log_dir:
            # 运行一次只生成第一个 token 的测试来测量 TTFT
            torch.cuda.synchronize()
            ttft_start_time = time.perf_counter()

            _ = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=self.sampling_params['do_sample'],
                top_k=self.sampling_params['top_k'],
                top_p=self.sampling_params['top_p'],
                temperature=self.sampling_params['temperature'],
            )

            torch.cuda.synchronize()
            ttft_end_time = time.perf_counter()
            ttft_s = max(ttft_end_time - ttft_start_time, 1e-6)

            print(f"[speed-log] TTFT measured: {ttft_s * 1000.0:.2f} ms", flush=True)

        # 记录开始时间（用于正常生成）
        torch.cuda.synchronize()
        arrival_time = time.perf_counter()

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.sampling_params['max_new_tokens'],
            do_sample=self.sampling_params['do_sample'],
            top_k=self.sampling_params['top_k'],
            top_p=self.sampling_params['top_p'],
            temperature=self.sampling_params['temperature'],
        )

        # 记录结束时间
        torch.cuda.synchronize()
        finished_time = time.perf_counter()

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # if </think> in the output, split it and return the last part

        # if "</think>" in output_text[0]:
        #     output_text[0] = output_text[0].split("</think>")[-1].strip()

        # res = output_text[0]

        ## split the response into prediction (without think) and detailed prediction (with think)

        if "</think>" in output_text[0]:
            prediction = output_text[0].split("</think>")[-1].strip()
            detailed_prediction = output_text[0].strip()
        else:
            prediction = output_text[0].strip()
            detailed_prediction = prediction

        # speed log (single)
        if self._speed_log_dir:
            try:
                # 计算输出 token 数量
                output_len = len(generated_ids_trimmed[0])
                output_len = max(0, output_len - 1)  # decode time 不包含第一个 token

                # 计算时间指标
                latency_s = max(finished_time - arrival_time, 1e-6)
                # 使用测量的 TTFT，如果没有测量则使用估算值
                # if ttft_s is None:
                #     ttft_s = latency_s * 0.15  # 估算值

                decode_time_s = max(latency_s - ttft_s, 1e-6)
                avg_decode_tps = (output_len / decode_time_s) if output_len > 0 else 0.0

                self._write_speed_log({
                    "ts": time.time(),
                    "type": "single",
                    "model": self.model_path,
                    "dataset": str(dataset) if dataset is not None else None,
                    "image_path": image_path,
                    "prompt_chars": len(text),
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

        prompts, image_paths = self.messages_to_promptimg(messages, dataset=dataset)

        batch_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
            for prompt, image_path in zip(prompts, image_paths)
        ]

        texts = self.processor.apply_chat_template(
            batch_messages, tokenize=False, add_generation_prompt=True
        )
        texts = [text + CUSTOM_PREFIX for text in texts]

        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # 记录开始时间
        torch.cuda.synchronize()
        arrival_time = time.perf_counter()

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.sampling_params['max_new_tokens'],
            do_sample=self.sampling_params['do_sample'],
            top_k=self.sampling_params['top_k'],
            top_p=self.sampling_params['top_p'],
            temperature=self.sampling_params['temperature'],
        )

        # 记录结束时间
        torch.cuda.synchronize()
        finished_time = time.perf_counter()
        total_latency_s = max(finished_time - arrival_time, 1e-6)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # # if </think> in the output, split it and return the last part
        # for i in range(len(output_texts)):
        #     if "</think>" in output_texts[i]:
        #         output_texts[i] = output_texts[i].split("</think>")[-1].strip()

        ## split the response into prediction (without think) and detailed prediction (with think)

        results_list = []
        for i in range(len(output_texts)):
            if "</think>" in output_texts[i]:
                prediction = output_texts[i].split("</think>")[-1].strip()
                detailed_prediction = output_texts[i].strip()
            else:
                prediction = output_texts[i].strip()
                detailed_prediction = prediction

            results_list.append({
                'prediction': prediction,
                'detailed_prediction': detailed_prediction,
            })

            # speed log (batch per-sample)
            if self._speed_log_dir:
                try:
                    # 计算输出 token 数量
                    output_len = len(generated_ids_trimmed[i])
                    output_len = max(0, output_len - 1)  # decode time 不包含第一个 token

                    # 对于 batch 模式，使用总时间作为每个样本的延迟（因为 batch 是并行处理的）
                    # TTFT 使用估算值
                    latency_s = total_latency_s
                    ttft_s = latency_s * 0.15  # 估算第一个 token 时间
                    decode_time_s = max(latency_s - ttft_s, 1e-6)
                    avg_decode_tps = (output_len / decode_time_s) if output_len > 0 else 0.0

                    self._write_speed_log({
                        "ts": time.time(),
                        "type": "batch",
                        "index": i,
                        "model": self.model_path,
                        "dataset": str(dataset) if dataset is not None else None,
                        "image_path": image_paths[i] if i < len(image_paths) else None,
                        "prompt_chars": len(texts[i]) if i < len(texts) else None,
                        "output_tokens": output_len,
                        "ttft_ms": (ttft_s * 1000.0) if ttft_s is not None else None,
                        "latency_ms": latency_s * 1000.0,
                        "avg_decode_tps": avg_decode_tps,
                    })
                except Exception as e:
                    print(f"[speed-log] batch record failed for idx={i}: {e}", flush=True)

        res = results_list

        return res
