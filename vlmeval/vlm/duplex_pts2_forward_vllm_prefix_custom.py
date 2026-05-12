import math
import torch
import torch.distributed as dist
import random
import numpy as np
from PIL import Image

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
import time
import json  # <-- 新增

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration
from vllm.model_executor.models import ModelRegistry

from .duplex_thinker_forward import DuplexThinkerForCausalLMForward

ModelRegistry.register_model("Qwen3ForCausalLM", DuplexThinkerForCausalLMForward)

VISUAL_BANDWIDTH=64

CUSTOM_PREFIX = os.environ.get("CUSTOM_PREFIX", "<think>\n\n</think>\n\n")
Temperature = float(os.getenv('TEMPERATURE', 0.01))
Layer = int(os.getenv('LAYER', 36))
Prompt = os.getenv('PROMPT', '')

def get_visual_message_tokens(num_tokens: int) -> List[str]:
    return [f'<im_msg-{i}>' for i in range(num_tokens)]

# ==== 速度统计工具函数（新增） ====
def _extract_times_from_output(req_output, default_latency_s: float):
    # 返回: ttft_s, latency_s, decode_time_s （单位：秒）
    res = {"ttft_s": None, "latency_s": default_latency_s, "decode_time_s": None}
    try:
        metrics = getattr(req_output, "metrics", None)
        if metrics is None:
            return res
        a = getattr(metrics, "arrival_time", None)
        f = getattr(metrics, "first_token_time", None)
        l = getattr(metrics, "last_token_time", None)
        fin = getattr(metrics, "finished_time", None)
        fs = getattr(metrics, "first_scheduled_time", None)

        if a is not None and f is not None:
            res["ttft_s"] = max(0.0, float(f) - float(a))

        if a is not None and fin is not None:
            res["latency_s"] = max(1e-6, float(fin) - float(a))
        elif a is not None and l is not None:
            res["latency_s"] = max(1e-6, float(l) - float(a))

        # use decode time as last - first
        if f is not None and l is not None:
            res["decode_time_s"] = max(1e-6, float(l) - float(f))
        elif f is not None and fin is not None:
            res["decode_time_s"] = max(1e-6, float(fin) - float(f))
    except Exception:
        pass
    return res

def _write_speed_log(speed_log_file: str, record: dict):
    if not speed_log_file:
        return
    try:
        with open(speed_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[speed-log] write failed: {e}", flush=True)
# ==== 速度统计工具函数（新增结束） ====

@torch.inference_mode()
def extract_batch_image_features(batch: List[Dict[str, Any]], perceiver_model, perceiver_processor, alignment_layer, device) -> List[torch.Tensor]:
    images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
    # p_prompt_template = 'Encode the image into {num_feat} tokens, including information related to the question. Here is the question: {question}'
    # visual_message_str = "".join(get_visual_message_tokens(VISUAL_BANDWIDTH))
    p_prompt_template = "{question}"

    p_texts = []
    for item in batch:
        p_prompt = p_prompt_template.format(question=item["question"])
        p_message = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': p_prompt}]}]
                    #  {'role': 'assistant', 'content': [{'type': 'text', 'text': visual_message_str}]}]
        p_texts.append(perceiver_processor.apply_chat_template(p_message, tokenize=False, add_generation_prompt=False))

    inputs = perceiver_processor(text=p_texts, images=images, return_tensors="pt", padding=True).to(device)
    outputs = perceiver_model(**inputs, output_hidden_states=True)
    last_hidden_states = outputs.hidden_states[-1]
    middle_hidden_states = outputs.hidden_states[Layer]
    print(f"now using layer {Layer} feature")
    # p_msg_token_ids = perceiver_processor.tokenizer.convert_tokens_to_ids(get_visual_message_tokens(VISUAL_BANDWIDTH))
    # p_msg_start_id, p_msg_end_id = p_msg_token_ids[0], p_msg_token_ids[-1]
    p_msg_start_id, p_msg_end_id = 151652, 151653

    batch_features = []
    for i in range(len(batch)):
        input_ids_sample = inputs.input_ids[i]
        start_indices = (input_ids_sample == p_msg_start_id).nonzero(as_tuple=True)[0]
        end_indices = (input_ids_sample == p_msg_end_id).nonzero(as_tuple=True)[0]
        if len(start_indices) == 0 or len(end_indices) == 0:
            raise ValueError(f"Visual message tokens not found in sample {i}")
        # features = last_hidden_states[i, start_indices[0]:end_indices[0] + 1, :]
        features = middle_hidden_states[i, start_indices[0]:end_indices[0] + 1, :]
        aligned_features = alignment_layer(features)
        batch_features.append(aligned_features)
    return batch_features

def batch_inference(batch_data: List[Dict[str, Any]], llm: LLM, thinker_tokenizer, perceiver_model, perceiver_processor, alignment_layer, device,
                    dataset: Any = None,              # <-- 新增：用于日志
    speed_log_file: str | None = None,  # <-- 新增：日志文件路径
    model_path: str | None = None,      # <-- 新增：用于日志
    log_type: str = "batch"             # <-- 新增："single"/"batch"
):
    print(f"\n--- Starting batch inference for {len(batch_data)} samples ---")

    print("Step 1: Extracting visual features...")
    # === 新增：计时特征提取 ===
    _ve_t0 = time.perf_counter()
    extracted_features = extract_batch_image_features(batch_data, perceiver_model, perceiver_processor, alignment_layer, device)
    _ve_t1 = time.perf_counter()
    visual_extract_time_s = max(_ve_t1 - _ve_t0, 0.0)
    # === 新增结束 ===

    print("Step 2: Preparing prompts for VLLM...")
    prompts_for_vllm = []
    # visual_message_str = "".join(get_visual_message_tokens(VISUAL_BANDWIDTH))
    # t_prompt_template = '{question} Image: ' + visual_message_str
    # t_prompt_template = '{question} {prompt}Image: ' + visual_message_str
    t_prompt_template = '{image}{question}'
    # t_prompt_template = '{question} Your response must be a single letter: A, B, C, or D. Image: ' + visual_message_str
    for idx, item in enumerate(batch_data):
        # t_prompt = t_prompt_template.format(question=item["question"])
        # t_prompt = t_prompt_template.format(question=item["question"], prompt=Prompt)
        extracted_feature_shape = extracted_features[idx].shape
        image_placeholder = "<|vision_start|>" + "<|image_pad|>" * (extracted_feature_shape[0] - 2) + "<|vision_end|>\n"
        t_prompt = t_prompt_template.format(image=image_placeholder, question=item["question"])
        message = [{"role": "user", "content": t_prompt}]
        final_prompt = thinker_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        ####### difference between _prefix
        final_prompt = final_prompt + CUSTOM_PREFIX
        #######
        prompts_for_vllm.append(final_prompt)

    print(f"shape of extracted_features: {[feat.shape for feat in extracted_features]}")

    print("Step 3: Assembling structured requests for VLLM...")
    requests = []
    for i in range(len(prompts_for_vllm)):
        extracted_features[i] = extracted_features[i].unsqueeze(1).to("cpu").to(torch.float32)
        print(f"Feature shape for sample {i}: {extracted_features[i].shape}")
        request = {
            "prompt": prompts_for_vllm[i],
            "multi_modal_data": {
                "image": extracted_features[i]
            }
        }
        requests.append(request)

    print("Step 4: Running batched generation with VLLM...")
    sampling_params = SamplingParams(temperature=Temperature, top_p=0.9, max_tokens=4096)
    print(f"using sampling params: {sampling_params}")

    if speed_log_file:
        sampling_params = SamplingParams(temperature=0.01, top_p=0.9, max_tokens=100)
        print(f"[speed-log] sampling_params changed to {sampling_params}", flush=True)
    # assert False
    # sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=30)
    _t0 = time.perf_counter()
    outputs = llm.generate(requests, sampling_params)
    _t1 = time.perf_counter()
    total_latency_s = max(_t1 - _t0, 1e-6)

    res = []
    for i, output in enumerate(outputs):
        if output.outputs:
            res.append(output.outputs[0].text.strip())
        else:
            res.append("No output generated.")

        # 速度日志（逐条）
        try:
            out_seq = output.outputs[0] if output.outputs else None
            token_ids = getattr(out_seq, "token_ids", []) or []
            output_len = (len(token_ids) if isinstance(token_ids, (list, tuple)) else 1)
            output_len = output_len - 1 # decode time 不包含第一个 token

            times = _extract_times_from_output(output, total_latency_s)
            ttft_s = times["ttft_s"]
            latency_s = times["latency_s"]
            decode_time_s = times["decode_time_s"]

            # === 新增：将特征提取耗时计入 TTFT 与 latency ===
            # 若无 TTFT 度量则从 0 开始叠加（确保至少包含特征提取耗时）
            ttft_s = (ttft_s or 0.0) + visual_extract_time_s
            # latency 如果为空则回退到 total_latency_s（仅包含 generate 段），再叠加特征提取
            latency_s = (latency_s or total_latency_s) + visual_extract_time_s
            # === 新增结束 ===

            if decode_time_s is None:
                decode_time_s = (latency_s - ttft_s) if (ttft_s is not None and latency_s >= ttft_s) else latency_s
            decode_time_s = max(decode_time_s, 1e-6)
            avg_decode_tps = (output_len / decode_time_s) if output_len > 0 else 0.0

            if speed_log_file:
                _write_speed_log(speed_log_file, {
                    "ts": time.time(),
                    "type": log_type,
                    "index": i if log_type == "batch" else None,
                    "model": model_path,
                    "dataset": str(dataset) if dataset is not None else None,
                    "image_path": batch_data[i]["image_path"] if i < len(batch_data) else None,
                    "prompt_chars": len(prompts_for_vllm[i]) if i < len(prompts_for_vllm) else None,
                    "output_tokens": output_len,
                    "ttft_ms": (ttft_s * 1000.0) if ttft_s is not None else None,
                    "latency_ms": latency_s * 1000.0,
                    "avg_decode_tps": avg_decode_tps,
                    "visual_extract_ms": visual_extract_time_s * 1000.0,
                })
        except Exception as e:
            print(f"[speed-log] record failed for idx={i}: {e}", flush=True)

    print(f"Generated {len(res)} responses.")
    return res

class DuplexThinkerS2ForwardvLLMPrefixCustom(Qwen2VLPromptMixin, BaseModel):

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

        self.model_path = model_path

        self.perceiver_device = "cuda:0"

        # SEPARATED_MODELS_DIR = Path("./separated_models")
        SEPARATED_MODELS_DIR = Path(model_path)
        PERCEIVER_PATH = SEPARATED_MODELS_DIR / "perceiver"
        THINKER_PATH = SEPARATED_MODELS_DIR / "thinker"
        ALIGNMENT_LAYER_PATH = SEPARATED_MODELS_DIR / "linear_align_dim.pth"
        VISUAL_BANDWIDTH = 64

        # 速度日志配置（新增）
        self._speed_log_dir = os.environ.get("speed_log_dir", "").strip()
        self._speed_log_file = None
        if self._speed_log_dir:
            os.makedirs(self._speed_log_dir, exist_ok=True)
            self._speed_log_file = os.path.join(self._speed_log_dir, f"{self.__class__.__name__}.jsonl")
            print(f"[speed-log] enabled. Writing to {self._speed_log_file}", flush=True)

        tensor_parallel_size = torch.cuda.device_count()
        if tensor_parallel_size == 0:
            raise ValueError("DualThinker with VLLM requires at least one CUDA GPU.")

        self.perceiver_processor = AutoProcessor.from_pretrained(PERCEIVER_PATH)
        self.perceiver_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            PERCEIVER_PATH,
            torch_dtype=torch.bfloat16,
        ).to(self.perceiver_device).eval()
        print("Perceiver model and processor loaded.")

        self.thinker_config = AutoConfig.from_pretrained(THINKER_PATH, trust_remote_code=True)
        # self.alignment_layer = torch.nn.Linear(
        #     self.perceiver_model.config.hidden_size,
        #     self.thinker_config.hidden_size,
        #     dtype=torch.bfloat16
        # ).to(self.perceiver_device).eval()
        # self.alignment_layer = torch.nn.Linear(
        #     self.perceiver_model.config.hidden_size,
        #     self.thinker_config.hidden_size,
        #     dtype=torch.bfloat16
        # ).to(self.perceiver_device).eval()
        print(f"hidden size of perceiver: {self.perceiver_model.config.hidden_size}, hidden size of thinker: {self.thinker_config.hidden_size}")
        # assert False
        self.alignment_layer = torch.nn.Sequential(
            torch.nn.Linear(self.perceiver_model.config.hidden_size, self.perceiver_model.config.hidden_size, dtype=torch.bfloat16),
            torch.nn.ReLU(),
            torch.nn.Linear(self.perceiver_model.config.hidden_size, self.thinker_config.hidden_size, dtype=torch.bfloat16),
        ).to(self.perceiver_device).eval()
        self.alignment_layer.load_state_dict(torch.load(ALIGNMENT_LAYER_PATH))

        print(f"weight of alignment layer: {self.alignment_layer[0].weight}, bias: {self.alignment_layer[0].bias}")
        print(f"wieght of alignment layer: {self.alignment_layer[2].weight}, bias: {self.alignment_layer[2].bias}")

        self.thinker_tokenizer = AutoTokenizer.from_pretrained(THINKER_PATH, padding_side="left")

        # get gpu count
        gpu_count = torch.cuda.device_count()

        self.llm = LLM(
            model=str(THINKER_PATH),
            trust_remote_code=True,
            tensor_parallel_size=gpu_count,
            gpu_memory_utilization=0.6,
            # enforce_eager=True,
            limit_mm_per_prompt={"image": 20480},
            max_num_seqs=1
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

        # return batch_inference(
        #     batch,
        #     self.llm,
        #     self.thinker_tokenizer,
        #     self.perceiver_model,
        #     self.perceiver_processor,
        #     self.alignment_layer,
        #     self.perceiver_device
        # )

        # only return one result
        return batch_inference(
            batch,
            self.llm,
            self.thinker_tokenizer,
            self.perceiver_model,
            self.perceiver_processor,
            self.alignment_layer,
            self.perceiver_device,
            dataset=dataset,                          # 新增
            speed_log_file=self._speed_log_file,      # 新增
            model_path=self.model_path,               # 新增
            log_type="single"                         # 新增
        )[0]

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
            self.perceiver_model,
            self.perceiver_processor,
            self.alignment_layer,
            self.perceiver_device,
            dataset=dataset,                          # 新增
            speed_log_file=self._speed_log_file,      # 新增
            model_path=self.model_path,               # 新增
            log_type="batch"                          # 新增
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
    # model = DuplexThinkerS2ForwardvLLMPrefixCustom(model_path="/path/to/home/workspace/duplex_think2/separated_models")
    model = DuplexThinkerS2ForwardvLLMPrefixCustom(model_path="/path/to/home/workspace/duplex_think2/separated_models_forward_mlp_pretrained_epo2_sft_1000_wsd_seed_47_epo2_finevision_4M_freeze")
    message = {"question": "What is the content of this image?", "image_path": "/path/to/home/workspace/duplex_think2/dummy_image.jpg"}
    response = model.generate_inner(message)
    print(response)
