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

# from vllm import SamplingParams

class DuplexThinkerS2(Qwen2VLPromptMixin, BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path="", use_custom_prompt=True, **kwargs):
        super().__init__(use_custom_prompt=use_custom_prompt)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        assert model_path is not None
        self.model_path = model_path

        tensor_parallel_size = torch.cuda.device_count()
        if tensor_parallel_size == 0:
            raise ValueError("DualThinker with VLLM requires at least one CUDA GPU.")

        print(f'Loading vllm model from {self.model_path}')

        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

        self.model = self.model.eval().cuda()

        self.kwargs = kwargs

        self.p_sampling_params = dict(do_sample=True, temperature=0.01, top_p=0.001, top_k=1, max_new_tokens=2048, repetition_penalty=1.0)
        self.t_sampling_params = dict(do_sample=True, temperature=0.01, top_p=0.95, top_k=20, max_new_tokens=32768)
        # self.t_sampling_params = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=32768)

    def generate_inner(self, message, dataset=None):
        print(f"Generating response for message: {message}", flush=True)

        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        # image = Image.open(image_path).convert('RGB')

        # image = self._resize_image_if_needed(image)

        msgs = [{'role': 'user', 'content': prompt}]

        print(f"Prepared message for generation: {msgs}", flush=True)

        res = self.model.chat(
            [image_path],
            [msgs],
            perceiver_generation_params=self.p_sampling_params,
            thinker_generation_params=self.t_sampling_params,
        )

        return res[0]

    def generate_batch_inner(self, messages, dataset=None):
        print(f"Generating batch with {len(messages)} messages.", flush=True)

        prompts, image_paths = self.messages_to_promptimgs(messages, dataset=dataset)

        # images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        # images = [self._resize_image_if_needed(image) for image in images]

        msgs = [{'role': 'user', 'content': prompt} for prompt in prompts]

        print(f"Prepared {len(msgs)} messages for batch generation.", flush=True)

        res = self.model.chat(
            image_paths,
            msgs,
            perceiver_generation_params=self.p_sampling_params,
            thinker_generation_params=self.t_sampling_params,
        )

        return res