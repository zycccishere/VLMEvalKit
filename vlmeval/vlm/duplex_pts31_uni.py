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

from vllm import SamplingParams

class DuplexThinkerS31Uni(Qwen2VLPromptMixin, BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path="", use_custom_prompt=True, **kwargs):
        super().__init__(use_custom_prompt=use_custom_prompt)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        self.VIDEO_LLM = True

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
        self.t_sampling_params = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=32768)

    def generate_inner(self, message, dataset=None):
        print(f"Generating response for message: {message}", flush=True)

        # check if message contains video
        num_videos = len([x for x in message if x['type'] == 'video'])
        num_images = len([x for x in message if x['type'] == 'image'])

        is_video = num_videos > 0
        is_image = num_images > 0

        if is_video and is_image:
            assert False, "Message contains both video and image, which is not supported."

        if is_video:
            prompt, video_path = self.message_to_promptvideo(message)
            msgs = [{'role': 'user', 'content': prompt}]
            print(f"Prepared message for video generation: {msgs}", flush=True)

            res = self.model.chat(
                messages=[
                    [{"role": "user", "content": [
                        {'type': 'video', 'video': video_path},
                        {'type': 'text', 'text': prompt}
                    ]}]
                ],
                perceiver_generation_params=self.p_sampling_params,
                thinker_generation_params=self.t_sampling_params,
            )
        else:
            prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
            msgs = [{'role': 'user', 'content': prompt}]

            print(f"Prepared message for generation: {msgs}", flush=True)

            res = self.model.chat(
                messages=[
                    [{"role": "user", "content": [
                        {'type': 'image', 'image': image_path},
                        {'type': 'text', 'text': prompt}
                    ]}]
                ],
                perceiver_generation_params=self.p_sampling_params,
                thinker_generation_params=self.t_sampling_params,
            )

        return res

    def generate_batch_inner(self, messages, dataset=None):
        print(f"Generating batch with {len(messages)} messages.", flush=True)

        # check if messages contain video
        num_videos = len([x for x in messages if x['type'] == 'video'])
        num_images = len([x for x in messages if x['type'] == 'image'])

        is_video = num_videos > 0
        is_image = num_images > 0

        if is_video and is_image:
            assert False, "Messages contain both video and image, which is not supported."

        if is_video:
            prompts, video_paths = self.message_to_promptvideos(messages)
            msgs = [{'role': 'user', 'content': prompt} for prompt in prompts]

            print(f"Prepared {len(msgs)} messages for video batch generation.", flush=True)

            res = self.model.chat(
                messages=[
                    [{"role": "user", "content": [
                        {'type': 'video', 'video': video_path},
                        {'type': 'text', 'text': prompt}
                    ]}] for prompt, video_path in zip(prompts, video_paths)
                ],
                perceiver_generation_params=self.p_sampling_params,
                thinker_generation_params=self.t_sampling_params,
            )
        else:
            prompts, image_paths = self.messages_to_promptimg(messages, dataset=dataset)
            msgs = [{'role': 'user', 'content': prompt} for prompt in prompts]

            print(f"Prepared {len(msgs)} messages for batch generation.", flush=True)

            res = self.model.chat(
                messages=[
                    [{"role": "user", "content": [
                        {'type': 'image', 'image': image_path},
                        {'type': 'text', 'text': prompt}
                    ]}] for prompt, image_path in zip(prompts, image_paths)
                ],
                perceiver_generation_params=self.p_sampling_params,
                thinker_generation_params=self.t_sampling_params,
            )

        return res
