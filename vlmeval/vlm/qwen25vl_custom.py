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

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
from qwen_vl_utils import process_vision_info

import os
Temperature = float(os.getenv('TEMPERATURE', 0.01))
HIGH_TEMP = os.getenv('HIGH_TEMP', 'False')

class Qwen25VLCustom(Qwen2VLPromptMixin, BaseModel):

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

        # self.sampling_params = dict(do_sample=True, temperature=Temperature, top_p=0.9, top_k=50, max_new_tokens=4096, repetition_penalty=1.0) # to be change back

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
        text = text + "<think>\n\n"

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.sampling_params['max_new_tokens'],
            do_sample=self.sampling_params['do_sample'],
            top_k=self.sampling_params['top_k'],
            top_p=self.sampling_params['top_p'],
            temperature=self.sampling_params['temperature'],
        )

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
        texts = [text + "<think>\n\n" for text in texts]

        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.sampling_params['max_new_tokens'],
            do_sample=self.sampling_params['do_sample'],
            top_k=self.sampling_params['top_k'],
            top_p=self.sampling_params['top_p'],
            temperature=self.sampling_params['temperature'],
        )

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

        res = results_list

        return res
