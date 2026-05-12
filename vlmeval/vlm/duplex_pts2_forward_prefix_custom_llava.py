import math
import torch
import torch.distributed as dist
import random
import numpy as np
from PIL import Image

from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE

from .qwen2_vl.prompt import Qwen2VLPromptMixin
from transformers import AutoModelForCausalLM


import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
import os
import time
import json

from transformers import AutoTokenizer, AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration

VISUAL_BANDWIDTH=64

CUSTOM_PREFIX = os.environ.get("CUSTOM_PREFIX", "<think>\n\n</think>\n\n")
Temperature = float(os.getenv('TEMPERATURE', 0.01))
Prompt = os.getenv('PROMPT', '')
MaxImageResolution = os.getenv('MAX_IMAGE_RESOLUTION', None)
if MaxImageResolution is not None:
    try:
        MaxImageResolution = int(MaxImageResolution)
    except ValueError:
        print(f"Warning: MAX_IMAGE_RESOLUTION={MaxImageResolution} is not a valid integer, ignoring it")
        MaxImageResolution = None

def get_visual_message_tokens(num_tokens: int) -> List[str]:
    return [f'<im_msg-{i}>' for i in range(num_tokens)]

@torch.inference_mode()
def extract_batch_image_features(batch: List[Dict[str, Any]], perceiver_model, perceiver_processor, alignment_layer, device) -> List[torch.Tensor]:
    images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
    p_prompt_template = "{question}"

    p_texts = []
    for item in batch:
        p_prompt = p_prompt_template.format(question=item["question"])
        p_message = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': p_prompt}]}]
        p_texts.append(perceiver_processor.apply_chat_template(p_message, tokenize=False, add_generation_prompt=False))

    inputs = perceiver_processor(text=p_texts, images=images, return_tensors="pt", padding=True).to(device)
    outputs = perceiver_model(**inputs, output_hidden_states=True)
    middle_hidden_states = outputs.hidden_states[-1]
    p_msg_start_id, p_msg_end_id = 151652, 151653

    batch_features = []
    perceiver_input_ids_list = []  # Store perceiver input_ids for later use
    for i in range(len(batch)):
        input_ids_sample = inputs.input_ids[i]
        perceiver_input_ids_list.append(input_ids_sample.tolist())

        start_indices = (input_ids_sample == p_msg_start_id).nonzero(as_tuple=True)[0]
        end_indices = (input_ids_sample == p_msg_end_id).nonzero(as_tuple=True)[0]
        if len(start_indices) == 0 or len(end_indices) == 0:
            raise ValueError(f"Visual message tokens not found in sample {i}")
        # Extract features including start and end tokens (end_indices[0] + 1 to include end token)
        features = middle_hidden_states[i, start_indices[0]:end_indices[0] + 1, :]
        aligned_features = alignment_layer(features)
        batch_features.append(aligned_features)

    return batch_features, perceiver_input_ids_list

def batch_inference(batch_data: List[Dict[str, Any]], thinker_model, thinker_tokenizer, perceiver_model, perceiver_processor, alignment_layer, device, speed_log_dir=None, dataset=None, image_paths=None, prompts_list_for_log=None):
    print(f"\n--- Starting batch inference for {len(batch_data)} samples ---")

    print("Step 1: Extracting visual features...")
    extracted_features, perceiver_input_ids_list = extract_batch_image_features(batch_data, perceiver_model, perceiver_processor, alignment_layer, device)

    print("Step 2: Preparing prompts for generation...")
    # Reference: ours_transformers_infer.py lines 157-193
    IMG_START_ID = 151652
    IMG_END_ID = 151653
    prompts_list = []
    for idx, item in enumerate(batch_data):
        # Calculate visual token count from perceiver input (same as reference)
        p_input_ids = perceiver_input_ids_list[idx]
        img_start_idx = p_input_ids.index(IMG_START_ID)
        img_end_idx = p_input_ids.index(IMG_END_ID)
        assert img_start_idx < img_end_idx

        # Calculate visual token count (excluding start and end tokens)
        num_visual_tokens = img_end_idx - img_start_idx - 1
        visual_tokens = (
            "<|vision_start|>"
            + "<|image_pad|>" * num_visual_tokens
            + "<|vision_end|>"
        )

        # Build prompt with visual tokens and question
        t_prompt = f"{visual_tokens}{item['question']}"
        message = [{"role": "user", "content": t_prompt}]
        final_prompt = thinker_tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Match reference implementation
        )
        ####### difference between _prefix
        final_prompt = final_prompt + CUSTOM_PREFIX
        #######
        prompts_list.append(final_prompt)

    # Use prompts_list for logging if prompts_list_for_log is not provided
    if prompts_list_for_log is None:
        prompts_list_for_log = prompts_list

    print(f"shape of extracted_features: {[feat.shape for feat in extracted_features]}")

    print("Step 3: Tokenizing prompts...")
    # Get model device
    model_device = next(thinker_model.parameters()).device
    model_dtype = next(thinker_model.parameters()).dtype

    # Tokenize all prompts
    inputs = thinker_tokenizer(
        prompts_list,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model_device)

    # Prepare image features for each sample
    # extracted_features[i] is [num_tokens, hidden_size]
    image_embeds_list = []
    for i in range(len(extracted_features)):
        feat = extracted_features[i].to(model_device).to(model_dtype)
        image_embeds_list.append(feat)
        print(f"Feature shape for sample {i}: {feat.shape}")

    print("Step 4: Running batched generation with transformers...")
    # Generate for each sample individually since we need to inject image features
    res = []

    # 如果启用测速日志，提示 max_new_tokens 已修改为 100
    if speed_log_dir:
        print(f"[speed-log] max_new_tokens changed to 100 for speed test", flush=True)

    # Helper function to write speed log
    def write_speed_log(record):
        if not speed_log_dir:
            return
        try:
            log_file = os.path.join(speed_log_dir, f"{thinker_model.__class__.__name__}.jsonl")
            os.makedirs(speed_log_dir, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[speed-log] write failed: {e}", flush=True)

    # Token IDs for vision tokens (from DuplexThinkerForCausalLMForward)
    visual_message_start_id = 151652
    visual_message_end_id = 151653
    visual_message_pad_id = 151655

    # Record batch start time
    batch_start_time = None
    if speed_log_dir:
        torch.cuda.synchronize()
        batch_start_time = time.perf_counter()

    for i in range(len(batch_data)):
        # Get input_ids and attention_mask for this sample
        input_ids = inputs.input_ids[i:i+1]  # [1, seq_len]
        attention_mask = inputs.attention_mask[i:i+1] if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None else None

        # print(f"input_ids: {input_ids}")
        # print(f"attention_mask: {attention_mask}")

        # Get input embeddings
        input_embeds = thinker_model.get_input_embeddings()(input_ids)  # [1, seq_len, hidden_size]

        # Prepare image_embeds for this sample
        image_embeds = image_embeds_list[i]  # [num_tokens, hidden_size]

        # Find visual message token positions and inject image features
        # Use the same approach as the reference implementation
        visual_mask = (input_ids[0] == visual_message_start_id) | \
                     (input_ids[0] == visual_message_end_id) | \
                     (input_ids[0] == visual_message_pad_id)

        if visual_mask.any():
            # Ensure we have the right number of image features
            num_visual_tokens = visual_mask.sum().item()
            if num_visual_tokens != image_embeds.shape[0]:
                raise ValueError(
                    f"Sample {i}: Mismatch between visual tokens ({num_visual_tokens}) "
                    f"and image features ({image_embeds.shape[0]})"
                )

            # Use masked_scatter to inject image features (same as reference implementation)
            mask_unsqueezed = visual_mask.unsqueeze(-1)  # [seq_len, 1]
            mask_expanded = mask_unsqueezed.expand_as(input_embeds[0])  # [seq_len, hidden_size]

            image_mask = mask_expanded.to(input_embeds.device)
            image_embeds = image_embeds.to(input_embeds.device).to(input_embeds.dtype)

            # masked_scatter fills in the order that True values appear in the mask
            input_embeds[0] = input_embeds[0].masked_scatter(image_mask, image_embeds)

        # Prepare generation kwargs
        # 如果启用测速日志，设置 max_new_tokens 为 100（参考 qwen25vl_custom_prefix_custom.py）
        max_new_tokens = 100 if speed_log_dir else 4096
        generation_kwargs = {
            "input_ids": None,  # Use inputs_embeds instead
            "inputs_embeds": input_embeds,
            "max_new_tokens": max_new_tokens,
            "temperature": Temperature,
            "top_p": 0.9,
            "do_sample": Temperature > 0,
        }

        # 如果启用测速日志，强制生成到 max_new_tokens（不让提前停止）
        if speed_log_dir:
            # 设置 min_new_tokens 等于 max_new_tokens，强制生成到最大长度
            # 这样即使遇到 EOS token，也会继续生成直到达到 min_new_tokens
            generation_kwargs["min_new_tokens"] = max_new_tokens

        # Add attention_mask if available
        if attention_mask is not None:
            generation_kwargs["attention_mask"] = attention_mask

        # Add pad_token_id if tokenizer has it
        if hasattr(thinker_tokenizer, 'pad_token_id') and thinker_tokenizer.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = thinker_tokenizer.pad_token_id
        elif hasattr(thinker_tokenizer, 'eos_token_id') and thinker_tokenizer.eos_token_id is not None:
            generation_kwargs["pad_token_id"] = thinker_tokenizer.eos_token_id

        # 如果启用测速日志，先运行一次 max_new_tokens=1 来测试 TTFT (single sample only)
        ttft_s = None
        if speed_log_dir and i == 0 and len(batch_data) == 1:
            torch.cuda.synchronize()
            ttft_start_time = time.perf_counter()

            # Create a copy of generation_kwargs with max_new_tokens=1 for TTFT test
            # TTFT test should not be forced to max tokens
            ttft_kwargs = generation_kwargs.copy()
            ttft_kwargs["max_new_tokens"] = 1
            ttft_kwargs.pop("min_new_tokens", None)  # Remove min_new_tokens for TTFT test

            _ = thinker_model.generate(**ttft_kwargs)

            torch.cuda.synchronize()
            ttft_end_time = time.perf_counter()
            ttft_s = max(ttft_end_time - ttft_start_time, 1e-6)
            print(f"[speed-log] TTFT measured: {ttft_s * 1000.0:.2f} ms", flush=True)

        # 记录开始时间（用于正常生成）
        torch.cuda.synchronize()
        arrival_time = time.perf_counter()

        # Generate with modified input embeddings
        with torch.inference_mode():
            generated_ids = thinker_model.generate(**generation_kwargs)

        # 记录结束时间
        torch.cuda.synchronize()
        finished_time = time.perf_counter()

        # Extract only the newly generated tokens (excluding the input prompt)
        # When using inputs_embeds, generated_ids may only contain newly generated tokens
        # or may contain input + generated tokens. We need to check the shape.
        input_length = input_ids.shape[1]
        generated_length = generated_ids.shape[1]

        if generated_length <= input_length:
            # generated_ids only contains newly generated tokens
            output_ids = generated_ids[0].tolist()
        else:
            # generated_ids contains input + generated tokens
            output_ids = generated_ids[0][input_length:].tolist()

        # Debug: Check if output_ids is empty
        if len(output_ids) == 0:
            print(f"[WARNING] Sample {i}: No tokens generated! input_length={input_length}, generated_ids.shape={generated_ids.shape}, generated_length={generated_length}")
            res.append("")
            continue

        # Decode the generated text
        # Match vLLM behavior: vLLM returns output.outputs[0].text which is the full generated text
        # We decode only the newly generated tokens (excluding input prompt)
        generated_text = thinker_tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        ).strip()

        # Debug: Check if generated_text is empty
        if not generated_text:
            print(f"[WARNING] Sample {i}: Generated text is empty! output_ids length={len(output_ids)}, first 10 tokens={output_ids[:10]}")

        res.append(generated_text)

        # speed log
        if speed_log_dir:
            try:
                # 计算输出 token 数量
                output_len = len(output_ids)
                output_len = max(0, output_len - 1)  # decode time 不包含第一个 token

                # 计算时间指标
                latency_s = max(finished_time - arrival_time, 1e-6)

                # 对于 batch 模式，如果是第一个样本且是单样本，使用测量的 TTFT
                # 否则使用估算值
                if ttft_s is None:
                    if len(batch_data) > 1:
                        # Batch mode: 使用总时间估算
                        ttft_s = latency_s * 0.15  # 估算第一个 token 时间
                    else:
                        ttft_s = latency_s * 0.15  # 单样本但未测量，使用估算值

                decode_time_s = max(latency_s - ttft_s, 1e-6)
                avg_decode_tps = (output_len / decode_time_s) if output_len > 0 else 0.0

                log_type = "single" if len(batch_data) == 1 else "batch"
                log_record = {
                    "ts": time.time(),
                    "type": log_type,
                    "index": i if len(batch_data) > 1 else None,
                    "model": str(thinker_model.__class__.__name__),
                    "dataset": str(dataset) if dataset is not None else None,
                    "image_path": image_paths[i] if image_paths and i < len(image_paths) else batch_data[i].get("image_path"),
                    "prompt_chars": len(prompts_list_for_log[i]) if prompts_list_for_log and i < len(prompts_list_for_log) else None,
                    "output_tokens": output_len,
                    "ttft_ms": (ttft_s * 1000.0) if ttft_s is not None else None,
                    "latency_ms": latency_s * 1000.0,
                    "avg_decode_tps": avg_decode_tps,
                }
                write_speed_log(log_record)
            except Exception as e:
                print(f"[speed-log] record failed for idx={i}: {e}", flush=True)

    print(f"Generated {len(res)} responses.")
    return res

class DuplexThinkerS2ForwardPrefixCustomLLaVA(Qwen2VLPromptMixin, BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path="", use_custom_prompt=True, **kwargs):
        super().__init__(use_custom_prompt=use_custom_prompt)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        with open("log.txt", "a") as f:
            f.write(f"Initializing DuplexThinkerS2 with model_path: {model_path}\n")
            if MaxImageResolution is not None:
                f.write(f"MAX_IMAGE_RESOLUTION from env: {MaxImageResolution} pixels\n")

        self.model_path = model_path

        tensor_parallel_size = torch.cuda.device_count()
        if tensor_parallel_size == 0:
            raise ValueError("DualThinker requires at least one CUDA GPU.")

        # 直接加载完整模型（包含 perceiver 和 thinker）
        print(f'Loading model from {model_path} using transformers...')
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()

        print("Model loaded successfully.")

        # 从环境变量读取最大分辨率，如果指定了则设置 image_processor 的 max_pixels
        if MaxImageResolution is not None:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'p_processor'):
                if hasattr(self.model.model.p_processor, 'image_processor'):
                    self.model.model.p_processor.image_processor.max_pixels = MaxImageResolution
                    print(f"Set image_processor.max_pixels to {MaxImageResolution} (from MAX_IMAGE_RESOLUTION env var)")
                else:
                    print(f"Warning: processor does not have image_processor attribute, MAX_IMAGE_RESOLUTION will be ignored")

        # 测速日志相关初始化
        self._speed_log_dir = os.environ.get("speed_log_dir", "").strip()
        if self._speed_log_dir:
            os.makedirs(self._speed_log_dir, exist_ok=True)
            print(f"[speed-log] enabled. Writing to {self._speed_log_dir}", flush=True)


    def generate_inner(self, message, dataset=None):
        print(f"Generating response for message: {message}", flush=True)
        print("----------------------------------------------")

        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        # 准备消息格式
        msgs = [{'role': 'user', 'content': prompt}]

        # 准备生成参数
        thinker_generation_params = {
            "max_new_tokens": 100 if self._speed_log_dir else 4096,
            "temperature": Temperature,
            "top_p": 0.9,
            "do_sample": Temperature > 0,
        }

        # 如果启用测速日志，强制生成到 max_new_tokens（不让提前停止）
        if self._speed_log_dir:
            thinker_generation_params["min_new_tokens"] = thinker_generation_params["max_new_tokens"]
            print(f"[speed-log] max_new_tokens changed to 100 for speed test", flush=True)

        # 记录开始时间（用于测速）
        ttft_s = None
        if self._speed_log_dir:
            torch.cuda.synchronize()
            arrival_time = time.perf_counter()

        # set max token to 1 to test ttft
        thinker_generation_params["max_new_tokens"] = 1
        with torch.inference_mode():
            responses = self.model.chat(
                [image_path],
                [msgs],
                thinker_generation_params=thinker_generation_params,
            )
        ttft_s = time.perf_counter() - arrival_time
        print(f"TTFT: {ttft_s * 1000.0:.2f} ms")

        thinker_generation_params["max_new_tokens"] = 100 if self._speed_log_dir else 4096

        # 调用模型的 chat 方法
        with torch.inference_mode():
            responses = self.model.chat(
                [image_path],
                [msgs],
                thinker_generation_params=thinker_generation_params,
            )

        # 记录结束时间（用于测速）
        if self._speed_log_dir:
            torch.cuda.synchronize()
            finished_time = time.perf_counter()
            latency_s = max(finished_time - arrival_time, 1e-6)

            # 写入测速日志
            try:
                response = responses[0] if responses else ""
                # 估算输出 token 数量（简单估算，实际应该从生成的 token 计算）
                output_len = max(len(response.split()) * 1.3, 0)  # 粗略估算
                output_len = max(0, int(output_len) - 1)  # decode time 不包含第一个 token

                # 估算 TTFT（如果没有测量）
                ttft_s = latency_s * 0.15  # 估算第一个 token 时间
                decode_time_s = max(latency_s - ttft_s, 1e-6)
                avg_decode_tps = (output_len / decode_time_s) if output_len > 0 else 0.0

                log_record = {
                    "ts": time.time(),
                    "type": "single",
                    "index": None,
                    "model": str(self.model.__class__.__name__),
                    "dataset": str(dataset) if dataset is not None else None,
                    "image_path": image_path,
                    "prompt_chars": len(prompt),
                    "output_tokens": output_len,
                    "ttft_ms": ttft_s * 1000.0,
                    "latency_ms": latency_s * 1000.0,
                    "avg_decode_tps": avg_decode_tps,
                }

                log_file = os.path.join(self._speed_log_dir, f"{self.model.__class__.__name__}.jsonl")
                os.makedirs(self._speed_log_dir, exist_ok=True)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[speed-log] record failed: {e}", flush=True)

        # 处理响应：移除 </think> 之前的内容
        response = responses[0] if responses else ""

        # 如果响应包含 </think>，只保留之后的内容
        # 注意：chat 方法已经处理了这部分，但为了兼容性，我们再次检查
        if "</think>" in response:
            parts = response.split("</think>")
            if len(parts) > 1:
                response = "</think>".join(parts[1:]).strip()

        return response

    def generate_batch_inner(self, messages, dataset=None):
        print(f"Generating batch with {len(messages)} messages.", flush=True)
        print("----------------------------------------------")

        prompts, image_paths = self.messages_to_promptimgs(messages, dataset=dataset)

        # images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        # images = [self._resize_image_if_needed(image) for image in images]

        batch = []
        for prompt, image_path in zip(prompts, image_paths):
            batch.append({"image_path": image_path, "question": prompt})
        print(f"Prepared {len(batch)} messages for batch generation.", flush=True)

        # Prepare prompts list for logging
        prompts_list_for_log = prompts if self._speed_log_dir else None

        return batch_inference(
            batch,
            self.thinker_model,
            self.thinker_tokenizer,
            self.perceiver_model,
            self.perceiver_processor,
            self.alignment_layer,
            self.perceiver_device,
            speed_log_dir=self._speed_log_dir,
            dataset=dataset,
            image_paths=image_paths,
            prompts_list_for_log=prompts_list_for_log
        )

        # msgs = [{'role': 'user', 'content': prompt} for prompt in prompts]

        # print(f"Prepared {len(msgs)} messages for batch generation.", flush=True)

        # res = self.model.chat(
        #     image_paths,
        #     msgs,
        #     perceiver_generation_params=self.p_sampling_params,
        #     thinker_generation_params=self.t_sampling_params,
        # )

        # return res


if __name__ == "__main__":
    # Example usage
    model = DuplexThinkerS2ForwardPrefixCustomLLaVA(model_path="/path/to/home/workspace/duplex_think2/separated_models_forward_mlp_pretrained_epo2_sft_1000_wsd_seed_47_epo2_finevision_4M_freeze")
    message = {"question": "What is the content of this image?", "image_path": "/path/to/home/workspace/duplex_think2/dummy_image.jpg"}
    response = model.generate_inner(message)
    print(response)
