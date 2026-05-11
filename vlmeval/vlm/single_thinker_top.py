import math
import torch
import torch.distributed as dist
import random
import numpy as np
from PIL import Image

from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE

from .single_thinker import VLLMSingleThinker
from .qwen2_vl.prompt import Qwen2VLPromptMixin

from vllm import SamplingParams

class VLLMSingleThinkerTop(Qwen2VLPromptMixin, BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path_1='', model_path_2='', use_custom_prompt=True, **kwargs):
        super().__init__(use_custom_prompt=use_custom_prompt)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        assert model_path_1 is not None
        assert model_path_2 is not None
        self.model_path_1 = model_path_1
        self.model_path_2 = model_path_2

        tensor_parallel_size = torch.cuda.device_count()
        if tensor_parallel_size == 0:
            raise ValueError("DualThinker with VLLM requires at least one CUDA GPU.")

        print(f'Loading vllm models from {self.model_path_1} and {self.model_path_2}')

        self.model = VLLMSingleThinker.from_pretrained(
            perceiver_model_name_or_path=self.model_path_1,
            thinker_model_name_or_path=self.model_path_2,
            tensor_parallel_size=tensor_parallel_size,
            **kwargs
        )

        self.kwargs = kwargs

        self.p_sampling_params = SamplingParams(temperature=0, max_tokens=2048)
        self.t_sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)

        self.min_image_size = 28

    def _resize_image_if_needed(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        if w >= self.min_image_size and h >= self.min_image_size:
            return image

        min_dim = min(w, h)
        scale_factor = self.min_image_size / min_dim

        new_w = int(w * scale_factor) + 1
        new_h = int(h * scale_factor) + 1

        resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        print(f"Warning: Resized image from ({w}, {h}) to ({new_w}, {new_h}) to meet model requirements.")

        return resized_image

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        image = Image.open(image_path).convert('RGB')

        image = self._resize_image_if_needed(image)

        msgs = [{'role': 'user', 'content': prompt}]

        res = self.model.chat(
            image=image,
            msgs=msgs,
            p_sampling_params=self.p_sampling_params,
            t_sampling_params=self.t_sampling_params
        )
        return res

    def generate_batch_inner(self, messages, dataset=None):
        prompts, image_paths = self.messages_to_promptimg(messages, dataset=dataset)

        images = [Image.open(image_path).convert('RGB') for image_path in image_paths]

        processed_images = [self._resize_image_if_needed(img) for img in images]

        msgs = [[{'role': 'user', 'content': prompt}] for prompt in prompts]

        res = self.model.generate(
            images=processed_images,
            msgs=msgs,
            p_sampling_params=self.p_sampling_params,
            t_sampling_params=self.t_sampling_params
        )

        return res