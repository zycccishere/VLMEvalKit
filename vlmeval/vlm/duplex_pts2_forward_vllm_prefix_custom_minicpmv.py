import math
import torch
import torch.distributed as dist
import random
import numpy as np
from PIL import Image
import threading
import time

from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE

# from .vllm_jointmodel import VLLMJointModel
from .qwen2_vl.prompt import Qwen2VLPromptMixin
from transformers import AutoModelForCausalLM

from vllm import SamplingParams

import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration
from vllm.model_executor.models import ModelRegistry

from .duplex_thinker_minicpmv import DuplexThinkerForCausalLMForward

ModelRegistry.register_model("Qwen3ForCausalLM", DuplexThinkerForCausalLMForward)

VISUAL_BANDWIDTH=64

# Profile统计：累计各步骤耗时
_profile_step1_time = 0.0
_profile_step2_time = 0.0
_profile_step3_time = 0.0
_profile_step4_time = 0.0

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

# Parse PERCEIVER_SKIP_LAYERS from environment variable
# Format: "start,end" (e.g., "1,1" to skip layer 1, or "2,5" to skip layers 2-5)
PERCEIVER_SKIP_LAYERS = None
skip_layers_str = os.getenv("PERCEIVER_SKIP_LAYERS", None)
if skip_layers_str:
    try:
        parts = skip_layers_str.split(",")
        if len(parts) == 2:
            start, end = int(parts[0].strip()), int(parts[1].strip())
            PERCEIVER_SKIP_LAYERS = [start, end]
            print(f"PERCEIVER_SKIP_LAYERS set from environment: {PERCEIVER_SKIP_LAYERS}")
        else:
            print(f"Warning: PERCEIVER_SKIP_LAYERS='{skip_layers_str}' has invalid format. Expected 'start,end'. Ignoring.")
    except ValueError as e:
        print(f"Warning: Failed to parse PERCEIVER_SKIP_LAYERS='{skip_layers_str}': {e}. Ignoring.")

def _env_truthy(name: str, default: str = "False") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y", "on")

def get_visual_message_tokens(num_tokens: int) -> List[str]:
    return [f'<im_msg-{i}>' for i in range(num_tokens)]

USE_SECOND_LAYER = os.environ.get("USE_SECOND_LAYER", "False") == "True"

@torch.inference_mode()
def extract_batch_image_features_single_gpu(batch: List[Dict[str, Any]], perceiver_model, perceiver_processor, alignment_layer, device) -> tuple[List[torch.Tensor], List[str]]:
    """
    Returns:
        batch_features: List of aligned visual features
        placeholders: List of placeholder strings for thinker (to be encoded by thinker tokenizer)
    """
    # MiniCPM-V token IDs for image markers in perceiver tokenizer
    IMG_START_ID = 151669
    IMG_END_ID = 151670
    IMG_SLICE_START_ID = 151679
    IMG_SLICE_END_ID = 151680
    IMG_ID_START_ID = 151681
    IMG_ID_END_ID = 151682

    images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
    question_start = _env_truthy("QUESTION_START")

    # prompts = []
    # images_list = []
    # for i, item in enumerate(batch):
    #     p_prompt = f"(<image>./</image>){item['question']}"
    #     # prompts.append(prompt)
    #     message = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': p_prompt}]}]
    #     prompts.append(perceiver_processor.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    #     images_list.append([images[i]])

    # inputs = perceiver_processor(
    #     prompts,
    #     images_list,
    #     max_slice_nums=9,
    #     use_image_id=True,
    #     return_tensors="pt",
    #     max_length=8192
    # )

    prompts = []
    images_list = []
    for i, item in enumerate(batch):
        if question_start:
            prompt = f"<|im_start|>user\n{item['question']}\n(<image>./</image>)"  # FIXME: Fail at multi-image samples
        else:
            prompt = f"<|im_start|>user\n(<image>./</image>)\n{item['question']}"  # FIXME: Fail at multi-image samples
        prompts.append(prompt)
        images_list.append([images[i]])
        print(f'sample-{i} prompts: {prompts}', flush=True)
        print(f'sample-{i} images_list: {images_list} image_path: {item["image_path"]}', flush=True)


    inputs = perceiver_processor(
        prompts,
        images_list,
        max_slice_nums=9,
        use_image_id=True,
        return_tensors="pt",
        max_length=8192
    )

    # # get input ids and decode for debug
    # input_ids = inputs.input_ids
    # print(f'input_ids: {input_ids}', flush=True)
    # exit()

    if os.environ.get("YTY_DEBUG", "False") == "True" or os.environ.get("ZYC_DEBUG", "False") == "True":
        print(f'perceiver_processor is {perceiver_processor}', flush=True)

    # print(f"keys in inputs before moving to device: {inputs.keys()}")

    def move_to_device(obj, target_device):
        if isinstance(obj, torch.Tensor):
            return obj.to(target_device)
        elif isinstance(obj, list):
            return [move_to_device(item, target_device) for item in obj]
        elif isinstance(obj, dict):
            return {k: move_to_device(v, target_device) for k, v in obj.items()}
        elif hasattr(obj, 'items'):  # Handle BatchFeature and similar dict-like objects
            for k in obj:
                obj[k] = move_to_device(obj[k], target_device)
            return obj
        else:
            return obj

    # Move all inputs to device (handles nested structures)
    for key in inputs:
        inputs[key] = move_to_device(inputs[key], device)
        if (os.environ.get("YTY_DEBUG", "False") == "True" or os.environ.get("ZYC_DEBUG", "False") == "True") and False:
            print(f'Process key: {key}', flush=True)
            if key == 'pixel_values':
                print(f'len(pixel_values): {len(inputs[key])}', flush=True)
                for i, lst in enumerate(inputs[key]):
                    print(f'{i}, len(lst): {len(lst)}', flush=True)
                    for j, img in enumerate(lst):
                        print(f'{j}, img shape: {img.shape}', flush=True)
                        print(f'{j}, img: {img}', flush=True)

    # if (os.environ.get("YTY_DEBUG", "False") == "True" or os.environ.get("ZYC_DEBUG", "False") == "True") and False:
    #     # Save debug images by reversing preprocessing
    #     debug_output_dir = "/path/to/vlmevalkit/debug_example_images"
    #     os.makedirs(debug_output_dir, exist_ok=True)

    #     pixel_values = inputs['pixel_values']  # List[List[Tensor]]
    #     tgt_sizes = inputs['tgt_sizes']  # List[Tensor] - [[H_patches, W_patches], ...]
    #     image_sizes = inputs['image_sizes']  # List[List[Tensor]] - [[original_H, original_W], ...]

    #     patch_size = 14
    #     norm_mean = 0.5
    #     norm_std = 0.5

    #     for batch_idx, (sample_imgs, sample_tgt_sizes) in enumerate(zip(pixel_values, tgt_sizes)):
    #         # sample_imgs: List[Tensor(3, 14, HW/14)] - multiple slices per image
    #         # sample_tgt_sizes: Tensor[[H_patches, W_patches], ...] for each slice

    #         for slice_idx, (img_tensor, tgt_size) in enumerate(zip(sample_imgs, sample_tgt_sizes)):
    #             # img_tensor shape: (3, patch_size, H*W/patch_size) = (3, 14, HW/14)
    #             # tgt_size: [H_patches, W_patches]

    #             H_patches = tgt_size[0].item()
    #             W_patches = tgt_size[1].item()
    #             H = H_patches * patch_size
    #             W = W_patches * patch_size

    #             num_patches = H_patches * W_patches

    #             print(f'Recovering image: batch={batch_idx}, slice={slice_idx}, '
    #                   f'tgt_size=[{H_patches}, {W_patches}], num_patches={num_patches}, '
    #                   f'reconstructed_shape=({H}, {W})', flush=True)

    #             # Move to CPU for processing
    #             img_cpu = img_tensor.cpu().float()

    #             # Reverse the reshape_by_patch function:
    #             # Original: [3, H, W] → unfold → [3, 196, num_patches] → reshape → [3, 14, 14, num_patches]
    #             #           → permute(0,1,3,2) → [3, 14, num_patches, 14] → reshape → [3, 14, num_patches*14]
    #             #
    #             # Reverse step-by-step:
    #             # Step 1: [3, 14, num_patches*14] → reshape → [3, 14, num_patches, 14]
    #             img_reshaped = img_cpu.reshape(3, patch_size, num_patches, patch_size)

    #             # Step 2: permute(0,1,3,2) is self-inverse → [3, 14, 14, num_patches]
    #             img_reshaped = img_reshaped.permute(0, 1, 3, 2)

    #             # Step 3: reshape → [3, 196, num_patches]
    #             img_reshaped = img_reshaped.reshape(3, patch_size * patch_size, num_patches)

    #             # Step 4: use fold to reverse unfold → [3, H, W]
    #             # fold expects input: [N, C × kernel_h × kernel_w, L]
    #             # We have: [3, 196, num_patches], reshape to [1, 3*196, num_patches]
    #             img_for_fold = img_reshaped.reshape(1, 3 * patch_size * patch_size, num_patches)

    #             img_reshaped = torch.nn.functional.fold(
    #                 img_for_fold,
    #                 output_size=(H, W),
    #                 kernel_size=(patch_size, patch_size),
    #                 stride=(patch_size, patch_size)
    #             )
    #             # Output: [1, 3, H, W] → squeeze → [3, H, W]
    #             img_reshaped = img_reshaped.squeeze(0)

    #             # Denormalize: pixel = normalized * std + mean
    #             img_denorm = img_reshaped * norm_std + norm_mean

    #             # Clip to [0, 1] and convert to uint8
    #             img_denorm = torch.clamp(img_denorm, 0.0, 1.0)
    #             img_np = (img_denorm * 255).byte().numpy()

    #             # Convert from (C, H, W) to (H, W, C) for PIL
    #             img_np = img_np.transpose(1, 2, 0)

    #             # Save as JPEG
    #             pil_img = Image.fromarray(img_np, mode='RGB')

    #             # Include shape info in filename
    #             orig_size = image_sizes[batch_idx][0] if len(image_sizes[batch_idx]) > 0 else None
    #             orig_h = orig_size[0].item() if orig_size is not None else 'unknown'
    #             orig_w = orig_size[1].item() if orig_size is not None else 'unknown'

    #             filename = f"batch{batch_idx}_slice{slice_idx}_origHW{orig_h}x{orig_w}_reconHW{H}x{W}.jpg"
    #             save_path = os.path.join(debug_output_dir, filename)
    #             pil_img.save(save_path, quality=95)
    #             print(f'Saved debug image to: {save_path}', flush=True)

    #     print(f'All debug images saved to: {debug_output_dir}', flush=True)
    #     exit()

    outputs = perceiver_model(data=None, **inputs, output_hidden_states=True)
    if USE_SECOND_LAYER:
        last_layer_hiddens = outputs.hidden_states[-2]
        print(f"Using second layer hidden states", flush=True)
    else:
        last_layer_hiddens = outputs.hidden_states[-1]

    image_processor = perceiver_processor.image_processor

    image_inputs = image_processor(images_list, return_tensors='pt', max_slice_nums=9)
    use_mlp_merger = ('grids' in image_inputs and image_inputs.grids is not None
                      and len(image_inputs.grids) > 0)

    # print(f"Now using connector: {'mlp_merger' if use_mlp_merger else 'resampler'}")

    batch_features = []
    placeholders = []

    for i in range(len(batch)):
        input_ids_sample = inputs.input_ids[i]
        # print(f"input_ids_sample: {input_ids_sample}")

        if os.getenv("FKC_DEBUG", "False") == "True" or os.getenv("YTY_DEBUG", "False") == "True" or os.getenv("ZYC_DEBUG", "False") == "True":
            try:
                image_inputs.grids[i][0]
            except Exception as e:
                print(f"sample-{i} before get slice: error: {e}", flush=True)
                print(f"sample-{i} before get slice: image_inputs.grids: {image_inputs.grids}", flush=True)
                print(f"sample-{i} before get slice: image_sizes: {image_inputs.image_sizes}", flush=True)
                print(f"sample-{i} before get slice: tgt_sizes: {image_inputs.tgt_sizes}", flush=True)
                print(f"sample-{i} before get slice: source_image_visual_tokens: {image_inputs.source_image_visual_tokens}", flush=True)
                print(f"sample-{i} before get slice: grids: {image_inputs.grids}", flush=True)
                print(f"sample-{i} before get slice: use_mlp_merger: {use_mlp_merger}", flush=True)
                print(f"sample-{i} before get slice: source_image_visual_tokens: {image_inputs.source_image_visual_tokens[i]}", flush=True)

        # Find image token positions in perceiver's input_ids
        img_start_indices = (input_ids_sample == IMG_START_ID).nonzero(as_tuple=True)[0].tolist()
        img_end_indices = (input_ids_sample == IMG_END_ID).nonzero(as_tuple=True)[0].tolist()
        slice_end_indices = (input_ids_sample == IMG_SLICE_END_ID).nonzero(as_tuple=True)[0].tolist()
        img_id_start_indices = (input_ids_sample == IMG_ID_START_ID).nonzero(as_tuple=True)[0].tolist()

        if len(img_start_indices) == 0 or len(img_end_indices) == 0:
            raise ValueError(f"No image tokens found in sample {i}")

        # Generate placeholder using get_slice_image_placeholder (for thinker)
        if use_mlp_merger:
            placeholder = image_processor.get_slice_image_placeholder(
                image_inputs.grids[i][0],
                image_idx=i,
                max_slice_nums=9,
                use_image_id=True,
                source_image_visual_tokens=image_inputs.source_image_visual_tokens[i][0],
                patch_visual_tokens=image_inputs.patch_visual_tokens[i][0]
            )
            # print(f"sample-{i} after get slice: placeholder: {placeholder}", flush=True)
        else:
            assert False
            placeholder = image_processor.get_slice_image_placeholder(
                image_inputs.image_sizes[i],
                image_idx=i,
                max_slice_nums=9,
                use_image_id=True
            )
        placeholder += '\n'

        if os.getenv('FKC_DEBUG', 'False') == 'True' or os.getenv('YTY_DEBUG', 'False') == 'True' or os.getenv("ZYC_DEBUG", "False") == "True":
            print(f"sample-{i} placeholder: {placeholder}", flush=True)

        # FIXME: id >= 10
        # replace <image_id>{id}</image_id> with <image_id><unk></image_id>
        # id is an arbitrary integer
        import re
        placeholder = re.sub(r'<image_id>(\d)</image_id>', r'<image_id><unk></image_id>', placeholder)
        # replace </slice>\n<slice> with </slice><unk><slice>
        placeholder = placeholder.replace('</slice>\n<slice>', '</slice><unk><slice>')

        if os.getenv('FKC_DEBUG', 'False') == 'True' or os.getenv('YTY_DEBUG', 'False') == 'True' or os.getenv("ZYC_DEBUG", "False") == "True":
            print(f"sample-{i} placeholder after replace: {placeholder}", flush=True)

        placeholders.append(placeholder)

        # Find the complete image range (including slices if present)
        img_start = img_start_indices[0]
        img_end = img_end_indices[0]

        # Check for slices after </image>
        image_slice_ends = [end for end in slice_end_indices if end > img_end]
        final_end = max(image_slice_ends) if image_slice_ends else img_end

        # Check for <image_id> before <image>
        actual_start = img_start
        for id_start in reversed(img_id_start_indices):
            if id_start < img_start:
                actual_start = id_start
                break

        # Extract features from the complete range
        features = last_layer_hiddens[i, actual_start:final_end + 1, :]
        aligned_features = alignment_layer(features)

        # print(f"sample-{i} start: {actual_start} end: {final_end + 1}, input_ids: {input_ids_sample[actual_start:final_end + 1]}")
        # print(f"decode: {final_end + 1 - actual_start} {perceiver_processor.tokenizer.decode(input_ids_sample[actual_start:final_end + 1])}")


        # Validate: count tokens in placeholder to ensure it matches feature count
        placeholder_token_count = (
            placeholder.count('<image>') + placeholder.count('</image>') +
            placeholder.count('<unk>') + placeholder.count('<slice>') +
            placeholder.count('</slice>') + placeholder.count('<image_id>') +
            placeholder.count('</image_id>')
        )

        image_token_count = placeholder.count('<image>')
        image_end_token_count = placeholder.count('</image>')
        unk_token_count = placeholder.count('<unk>')
        slice_token_count = placeholder.count('<slice>')
        slice_end_token_count = placeholder.count('</slice>')
        image_id_token_count = placeholder.count('<image_id>')
        image_id_end_token_count = placeholder.count('</image_id>')

        feature_count = aligned_features.shape[0]
        if placeholder_token_count != feature_count:
            print(f"WARNING: Token count mismatch for sample {i}!")

        # print(f"  Placeholder tokens: {placeholder_token_count}  Extracted features: {feature_count}  image_token_count: {image_token_count}  image_end_token_count: {image_end_token_count}  unk_token_count: {unk_token_count}  slice_token_count: {slice_token_count}  slice_end_token_count: {slice_end_token_count}  image_id_token_count: {image_id_token_count}  image_id_end_token_count: {image_id_end_token_count}   Placeholder: {placeholder.replace('<unk>', '!')} patch_visual_tokens: {image_inputs.patch_visual_tokens[i][0]} source_image_visual_tokens: {image_inputs.source_image_visual_tokens[i][0]}  grid: {image_inputs.grids[i][0]}")

        batch_features.append(aligned_features)

    return batch_features, placeholders


@torch.inference_mode()
def extract_batch_image_features(batch: List[Dict[str, Any]], perceiver_models, perceiver_processor, alignment_layers, num_gpus: int) -> tuple[List[torch.Tensor], List[str]]:
    """
    多 GPU 并行提取视觉特征

    Args:
        batch: 批次数据列表
        perceiver_models: 每个 GPU 上的 perceiver 模型列表
        perceiver_processor: 处理器（共享）
        alignment_layers: 每个 GPU 上的 alignment layer 列表
        num_gpus: GPU 数量

    Returns:
        batch_features: List of aligned visual features
        placeholders: List of placeholder strings for thinker
    """
    if num_gpus == 1:
        # 单 GPU 情况，直接调用单 GPU 版本
        return extract_batch_image_features_single_gpu(
            batch, perceiver_models[0], perceiver_processor, alignment_layers[0], "cuda:0"
        )

    # 多 GPU 情况：将 batch 分割到不同 GPU
    batch_size = len(batch)
    samples_per_gpu = (batch_size + num_gpus - 1) // num_gpus  # 向上取整

    # 分割 batch
    batch_splits = []
    for i in range(num_gpus):
        start_idx = i * samples_per_gpu
        end_idx = min((i + 1) * samples_per_gpu, batch_size)
        if start_idx < batch_size:
            batch_splits.append(batch[start_idx:end_idx])
        else:
            batch_splits.append([])

    # 在每个 GPU 上并行处理（使用多线程）
    all_features = [None] * num_gpus
    all_placeholders = [None] * num_gpus
    threads = []

    def process_gpu(gpu_id, gpu_batch, device):
        """在指定 GPU 上处理 batch 的线程函数"""
        if len(gpu_batch) == 0:
            all_features[gpu_id] = []
            all_placeholders[gpu_id] = []
            return

        # 在对应 GPU 上处理
        gpu_features, gpu_placeholders = extract_batch_image_features_single_gpu(
            gpu_batch,
            perceiver_models[gpu_id],
            perceiver_processor,
            alignment_layers[gpu_id],
            device
        )
        all_features[gpu_id] = gpu_features
        all_placeholders[gpu_id] = gpu_placeholders

    # 启动所有 GPU 的处理线程
    for gpu_id in range(num_gpus):
        if len(batch_splits[gpu_id]) == 0:
            all_features[gpu_id] = []
            all_placeholders[gpu_id] = []
            continue

        device = f"cuda:{gpu_id}"
        gpu_batch = batch_splits[gpu_id]

        thread = threading.Thread(
            target=process_gpu,
            args=(gpu_id, gpu_batch, device)
        )
        thread.start()
        threads.append(thread)

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    result_features = []
    result_placeholders = []
    for gpu_id in range(num_gpus):
        if all_features[gpu_id] is not None:
            for feat in all_features[gpu_id]:
                result_features.append(feat.to("cuda:0"))
            result_placeholders.extend(all_placeholders[gpu_id])

    return result_features, result_placeholders

def batch_inference(batch_data: List[Dict[str, Any]], llm: LLM, thinker_tokenizer, perceiver_models, perceiver_processor, alignment_layers, num_gpus: int):
    global _profile_step1_time, _profile_step2_time, _profile_step3_time, _profile_step4_time
    zyc_debug = os.getenv("ZYC_DEBUG", "False") == "True"

    print(f"\n--- Starting batch inference for {len(batch_data)} samples ---")

    print("Step 1: Extracting visual features (parallel on {} GPUs)...".format(num_gpus))
    step1_start = time.perf_counter()
    extracted_features, placeholders = extract_batch_image_features(batch_data, perceiver_models, perceiver_processor, alignment_layers, num_gpus)
    step1_end = time.perf_counter()
    step1_duration = step1_end - step1_start
    _profile_step1_time += step1_duration

    print("Step 2: Preparing prompts for VLLM...")
    step2_start = time.perf_counter()
    prompts_for_vllm = []
    t_prompt_template = '{image}\n{question}'

    for idx, item in enumerate(batch_data):
        # Use the placeholder generated by get_slice_image_placeholder
        # This placeholder contains MiniCPM-V token structure:
        # e.g., "<image_id>0</image_id><image><unk>*N</image><slice><unk>*M</slice>...\n"
        #
        # IMPORTANT: Keep MiniCPM-V tokens as-is, do NOT convert to Qwen tokens
        # The thinker tokenizer will encode these MiniCPM-V tokens directly
        placeholder = placeholders[idx]

        t_prompt = t_prompt_template.format(image=placeholder, question=item["question"])
        message = [{"role": "user", "content": t_prompt}]
        final_prompt = thinker_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        ####### difference between _prefix
        final_prompt = final_prompt + CUSTOM_PREFIX
        #######
        prompts_for_vllm.append(final_prompt)
        if zyc_debug:
            print(f"sample-{idx} thinker_prompt: {final_prompt}", flush=True)
    step2_end = time.perf_counter()
    step2_duration = step2_end - step2_start
    _profile_step2_time += step2_duration

    # print(f"shape of extracted_features: {[feat.shape for feat in extracted_features]}")
    # print(f"placeholders (first 200 chars): {[p[:200] for p in placeholders]}")

    print("Step 3: Assembling structured requests for VLLM...")
    step3_start = time.perf_counter()
    requests = []
    for i in range(len(prompts_for_vllm)):
        extracted_features[i] = extracted_features[i].unsqueeze(1).to("cpu").to(torch.float32)
        # print(f"Feature shape for sample {i}: {extracted_features[i].shape}")
        request = {
            "prompt": prompts_for_vllm[i],
            "multi_modal_data": {
                "image": extracted_features[i]
            }
        }
        requests.append(request)
    step3_end = time.perf_counter()
    step3_duration = step3_end - step3_start
    _profile_step3_time += step3_duration

    print("Step 4: Running batched generation with VLLM...")
    step4_start = time.perf_counter()
    sampling_params = SamplingParams(temperature=Temperature, top_p=0.9, max_tokens=4096)
    print(f"using sampling params: {sampling_params}")

    outputs = llm.generate(requests, sampling_params)
    if zyc_debug:
        print(f"thinker outputs: {len(outputs)}", flush=True)
    step4_end = time.perf_counter()
    step4_duration = step4_end - step4_start
    _profile_step4_time += step4_duration

    res = []
    for i, output in enumerate(outputs):
        if output.outputs:
            text = output.outputs[0].text.strip()
            res.append(text)
        else:
            text = "No output generated."
            res.append(text)
        if zyc_debug:
            print(f"sample-{i} thinker_output: {text}", flush=True)

    # 打印累计性能统计
    # print(f"\n=== Batch Inference Profile Statistics (累计) ===")
    # print(f"Step 1 (Extracting visual features): {_profile_step1_time:.4f}s")
    # print(f"Step 2 (Preparing prompts for VLLM): {_profile_step2_time:.4f}s")
    # print(f"Step 3 (Assembling structured requests): {_profile_step3_time:.4f}s")
    # print(f"Step 4 (Running batched generation with VLLM): {_profile_step4_time:.4f}s")
    # print(f"Total (Steps 1-4): {_profile_step1_time + _profile_step2_time + _profile_step3_time + _profile_step4_time:.4f}s")
    # print(f"================================================\n")

    print(f"Generated {len(res)} responses.")
    return res

class DuplexThinkerS2ForwardvLLMPrefixCustomMiniCPMV(Qwen2VLPromptMixin, BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path="", use_custom_prompt=True, **kwargs):
        super().__init__(use_custom_prompt=use_custom_prompt)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        # with open("log.txt", "a") as f:
        #     f.write(f"Initializing DuplexThinkerS2vLLM with model_path: {model_path}\n")
        #     if MaxImageResolution is not None:
        #         f.write(f"MAX_IMAGE_RESOLUTION from env: {MaxImageResolution} pixels\n")

        self.model_path = model_path

        # SEPARATED_MODELS_DIR = Path("./separated_models")
        SEPARATED_MODELS_DIR = Path(model_path)
        PERCEIVER_PATH = SEPARATED_MODELS_DIR / "perceiver"
        THINKER_PATH = SEPARATED_MODELS_DIR / "thinker"
        ALIGNMENT_LAYER_PATH = SEPARATED_MODELS_DIR / "linear_align_dim.pth"
        VISUAL_BANDWIDTH = 64

        tensor_parallel_size = torch.cuda.device_count()
        if tensor_parallel_size == 0:
            raise ValueError("DualThinker with VLLM requires at least one CUDA GPU.")

        self.num_gpus = tensor_parallel_size
        print(f"Detected {self.num_gpus} GPUs for parallel perceiver inference")

        self.perceiver_processor = AutoProcessor.from_pretrained(PERCEIVER_PATH, trust_remote_code=True)
        # 从环境变量读取最大分辨率，如果指定了则设置 image_processor 的 max_pixels
        if MaxImageResolution is not None:
            if hasattr(self.perceiver_processor, 'image_processor'):
                self.perceiver_processor.image_processor.max_pixels = MaxImageResolution
                print(f"Set image_processor.max_pixels to {MaxImageResolution} (from MAX_IMAGE_RESOLUTION env var)")
            else:
                print(f"Warning: processor does not have image_processor attribute, MAX_IMAGE_RESOLUTION will be ignored")

        # Load MiniCPM-V model
        from transformers import AutoModelForCausalLM

        # Load config and set perceiver_skip_layers if specified
        perceiver_config = AutoConfig.from_pretrained(PERCEIVER_PATH, trust_remote_code=True)
        if PERCEIVER_SKIP_LAYERS is not None:
            perceiver_config.perceiver_skip_layers = PERCEIVER_SKIP_LAYERS
            print(f"Setting perceiver_skip_layers to {PERCEIVER_SKIP_LAYERS} in config")

        base_perceiver_model = AutoModelForCausalLM.from_pretrained(
            PERCEIVER_PATH,
            config=perceiver_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        # MiniCPM-V hidden size is in config.hidden_size
        perceiver_hidden_size = base_perceiver_model.config.hidden_size

        # 将 perceiver_model 复制到所有 GPU
        self.perceiver_models = []
        for gpu_id in range(self.num_gpus):
            device = f"cuda:{gpu_id}"
            if gpu_id == 0:
                # 第一个 GPU 使用已加载的模型
                model = base_perceiver_model.to(device).eval()
            else:
                # 其他 GPU 重新加载模型（避免共享内存问题）
                # 使用相同的 config（包含 perceiver_skip_layers）
                model = AutoModelForCausalLM.from_pretrained(
                    PERCEIVER_PATH,
                    config=perceiver_config,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                ).to(device).eval()
            self.perceiver_models.append(model)
            print(f"Perceiver model loaded on {device}")

        print("All perceiver models loaded on all GPUs.")

        self.thinker_config = AutoConfig.from_pretrained(THINKER_PATH, trust_remote_code=True)
        print(f"hidden size of perceiver: {perceiver_hidden_size}, hidden size of thinker: {self.thinker_config.hidden_size}")

        # 将 alignment_layer 复制到所有 GPU
        self.alignment_layers = []
        alignment_state_dict = torch.load(ALIGNMENT_LAYER_PATH)
        for gpu_id in range(self.num_gpus):
            device = f"cuda:{gpu_id}"
            alignment_layer = torch.nn.Sequential(
                torch.nn.Linear(perceiver_hidden_size, perceiver_hidden_size, dtype=torch.bfloat16),
                torch.nn.ReLU(),
                torch.nn.Linear(perceiver_hidden_size, self.thinker_config.hidden_size, dtype=torch.bfloat16),
            ).to(device).eval()
            alignment_layer.load_state_dict(alignment_state_dict)
            self.alignment_layers.append(alignment_layer)
            print(f"Alignment layer loaded on {device}")

        # 为了向后兼容，保留单个模型的引用（指向第一个 GPU）
        self.perceiver_model = self.perceiver_models[0]
        self.alignment_layer = self.alignment_layers[0]

        print(f"weight of alignment layer: {self.alignment_layer[0].weight}, bias: {self.alignment_layer[0].bias}")
        print(f"wieght of alignment layer: {self.alignment_layer[2].weight}, bias: {self.alignment_layer[2].bias}")

        # self.thinker_tokenizer = AutoTokenizer.from_pretrained(THINKER_PATH, padding_side="left")
        self.thinker_tokenizer = AutoTokenizer.from_pretrained("/path/to/checkpoints/siglip_ours/duplex_qwen3_tokenizer_w_minicpm", padding_side="left")

        # get gpu count
        gpu_count = torch.cuda.device_count()

        self.llm = LLM(
            model=str(THINKER_PATH),
            trust_remote_code=True,
            tensor_parallel_size=gpu_count,
            gpu_memory_utilization=0.6,
            # enforce_eager=True,
            limit_mm_per_prompt={"image": 20480},
            max_num_seqs=8
        )

        # print(f'Loading vllm model from {self.model_path}')

        # self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

        # self.model = self.model.eval().cuda()

        # self.kwargs = kwargs

        # self.p_sampling_params = dict(do_sample=True, temperature=0.01, top_p=0.001, top_k=1, max_new_tokens=2048, repetition_penalty=1.0)
        # self.t_sampling_params = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=32768)


    def generate_inner(self, message, dataset=None):
        print(f"Generating response for message: {message}", flush=True)
        print("----------------------------------------------")
        # assert False

        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        # image = Image.open(image_path).convert('RGB')

        # image = self._resize_image_if_needed(image)

        # msgs = [{'role': 'user', 'content': prompt}]

        # print(f"Prepared message for generation: {msgs}", flush=True)

        # res = self.model.chat(
        #     [image_path],
        #     [msgs],
        #     perceiver_generation_params=self.p_sampling_params,
        #     thinker_generation_params=self.t_sampling_params,
        # )

        batch = [
            {"image_path": image_path, "question": prompt}
        ]

        return batch_inference(
            batch,
            self.llm,
            self.thinker_tokenizer,
            self.perceiver_models,
            self.perceiver_processor,
            self.alignment_layers,
            self.num_gpus
        )

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
        return batch_inference(
            batch,
            self.llm,
            self.thinker_tokenizer,
            self.perceiver_models,
            self.perceiver_processor,
            self.alignment_layers,
            self.num_gpus
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
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="测试 DuplexThinkerS2ForwardvLLMPrefixCustomMiniCPMV 模型")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型路径（包含 perceiver、thinker 和 linear_align_dim.pth 的目录）"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=False,
        help="测试图像路径（单图测试）"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What is the content of this image?",
        help="测试问题（默认: 'What is the content of this image?'）"
    )
    parser.add_argument(
        "--batch_mode",
        action="store_true",
        help="启用批量测试模式（需要提供多个图像路径）"
    )
    parser.add_argument(
        "--image_paths",
        type=str,
        nargs="+",
        help="批量测试的图像路径列表（需要 --batch_mode）"
    )
    parser.add_argument(
        "--questions",
        type=str,
        nargs="+",
        help="批量测试的问题列表（可选，默认使用 --question）"
    )

    args = parser.parse_args()

    # 验证模型路径
    from pathlib import Path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"错误: 模型路径不存在: {model_path}", file=sys.stderr)
        sys.exit(1)

    perceiver_path = model_path / "perceiver"
    thinker_path = model_path / "thinker"
    alignment_path = model_path / "linear_align_dim.pth"

    if not perceiver_path.exists():
        print(f"错误: Perceiver 路径不存在: {perceiver_path}", file=sys.stderr)
        sys.exit(1)
    if not thinker_path.exists():
        print(f"错误: Thinker 路径不存在: {thinker_path}", file=sys.stderr)
        sys.exit(1)
    if not alignment_path.exists():
        print(f"错误: Alignment layer 文件不存在: {alignment_path}", file=sys.stderr)
        sys.exit(1)

    print(f"正在加载模型: {model_path}")
    print(f"  - Perceiver: {perceiver_path}")
    print(f"  - Thinker: {thinker_path}")
    print(f"  - Alignment layer: {alignment_path}")
    print("-" * 60)

    try:
        # 初始化模型
        model = DuplexThinkerS2ForwardvLLMPrefixCustomMiniCPMV(model_path=str(model_path))
        print("模型加载成功！\n")

        if args.batch_mode:
            # 批量测试模式
            if not args.image_paths:
                print("错误: 批量模式需要提供 --image_paths", file=sys.stderr)
                sys.exit(1)

            image_paths = args.image_paths
            questions = args.questions if args.questions else [args.question] * len(image_paths)

            if len(questions) != len(image_paths):
                print(f"警告: 问题数量 ({len(questions)}) 与图像数量 ({len(image_paths)}) 不匹配，将重复使用最后一个问题", file=sys.stderr)
                if len(questions) < len(image_paths):
                    questions.extend([questions[-1]] * (len(image_paths) - len(questions)))
                else:
                    questions = questions[:len(image_paths)]

            # 验证图像路径
            for img_path in image_paths:
                if not Path(img_path).exists():
                    print(f"警告: 图像路径不存在: {img_path}", file=sys.stderr)

            print(f"批量测试模式: {len(image_paths)} 个样本")
            print("-" * 60)

            messages = []
            for img_path, q in zip(image_paths, questions):
                messages.append({
                    "question": q,
                    "image_path": img_path
                })
                print(f"样本 {len(messages)}: 图像={img_path}, 问题={q}")

            print("-" * 60)
            print("开始批量推理...\n")

            responses = model.generate_batch_inner(messages)

            print("\n" + "=" * 60)
            print("批量推理结果:")
            print("=" * 60)
            for i, (msg, resp) in enumerate(zip(messages, responses), 1):
                print(f"\n样本 {i}:")
                print(f"  图像: {msg['image_path']}")
                print(f"  问题: {msg['question']}")
                print(f"  回答: {resp}")

        else:
            # 单图测试模式
            if not args.image_path:
                print("错误: 单图模式需要提供 --image_path", file=sys.stderr)
                sys.exit(1)

            image_path = Path(args.image_path)
            if not image_path.exists():
                print(f"错误: 图像路径不存在: {image_path}", file=sys.stderr)
                sys.exit(1)

            print(f"单图测试模式")
            print(f"  图像: {image_path}")
            print(f"  问题: {args.question}")
            print("-" * 60)
            print("开始推理...\n")

            message = [
                {"type": "image", "value": image_path},
                {"type": "text", "value": args.question}
            ]

            responses = model.generate_inner(message)

            print("\n" + "=" * 60)
            print("推理结果:")
            print("=" * 60)
            if isinstance(responses, list) and len(responses) > 0:
                print(f"\n回答: {responses[0]}")
            else:
                print(f"\n回答: {responses}")

        print("\n测试完成！")

    except Exception as e:
        print(f"\n错误: 测试过程中发生异常: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
