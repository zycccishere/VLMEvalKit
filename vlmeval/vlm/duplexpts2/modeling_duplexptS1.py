import torch
import argparse

from PIL import Image
from typing import Callable, Optional, Union
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import List, Dict, Union
from transformers import Qwen2PreTrainedModel
from configuration_duplexptS1 import DuplexPTS1Config


# text, image => ids + features => out_ids
# text, image => ids + features => perceiver out_ids => thinker input_ids => thinker out_ids
# 实现 LLM 类

from transformers.utils import auto_docstring, can_return_tuple
from transformers.cache_utils import Cache, DynamicCache

class DuplexPTS1PreTrainedModel(Qwen2PreTrainedModel):
    config_class = DuplexPTS1Config


# 继承 transformers pretrained_causal_lm
class DuplexPTS1Model(DuplexPTS1PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.perceiver = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.perceiver_name_or_path,
            # torch_dtype="auto",
            # device_map="auto",
        )
        self.p_processor = AutoProcessor.from_pretrained(
            self.config.perceiver_name_or_path)
        self.p_processor.tokenizer.padding_side = "left"

        self.thinker = AutoModelForCausalLM.from_pretrained(
            self.config.thinker_name_or_path,
            # torch_dtype="auto",
            # device_map="auto",
        )
        self.t_tokenizer = AutoTokenizer.from_pretrained(
            self.config.thinker_name_or_path, padding_side="left")

        self.forward_p = False
        self.forward_t = True
        assert self.forward_p ^ self.forward_t

    def chat(self, image, msgs, *args, **kwargs):
        assert len(msgs) == 1 and msgs[0]['role'] == 'user'
        assert args == ()
        assert 'max_new_tokens' not in kwargs

        return [x[0] for x in self.generate([image], [msgs], *args, **kwargs)]

    def generate(self, images: List[Union[str, Image.Image]], msgs: List[List[Dict[str, str]]], *args, **kwargs) -> List[str]:
        assert len(images) == len(msgs)
        assert args == ()
        assert 'max_new_tokens' not in kwargs

        p_prompt_template = 'Please describe the image in great detail, including information related to the question. Here is the question: {question}'
        questions = []
        p_images = []
        p_texts = []

        for i in range(len(images)):
            image = images[i]
            msg_list = msgs[i]

            if not (len(msg_list) == 1 and msg_list[0]['role'] == 'user'):
                raise ValueError(
                    f"Each message list must contain a single user dictionary. Error at index {i}.")

            pil_image = Image.open(image).convert(
                "RGB") if isinstance(image, str) else image
            p_images.append(pil_image)

            question = msg_list[0]['content']
            questions.append(question)

            p_message = [{'role': 'user', 'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': p_prompt_template.format(
                    question=question)}
            ]}]
            print(f'P-Message-{i}: {p_message}')
            p_texts.append(self.p_processor.apply_chat_template(
                p_message, tokenize=False, add_generation_prompt=True))

        print(f'{p_texts=}')
        inputs = self.p_processor(
            text=p_texts,
            images=p_images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # print('Token IDs of perceiver inputs', inputs['input_ids'].tolist())
        # print('Tokens of perceiver inputs', [self.t_tokenizer.convert_ids_to_tokens(ids) for ids in inputs['input_ids']])

        # print(inputs['attention_mask'].tolist())
        # inputs['attention_mask'][:, :] = 1
        # print(inputs['attention_mask'].tolist())

        perceiver_generation_params = kwargs.get('perceiver_generation_params', {})
        perceiver_generation_params['max_new_tokens'] = perceiver_generation_params.get(
            'max_new_tokens', 128)

        generated_ids_p = self.perceiver.generate(
            **inputs,
            **perceiver_generation_params,
            pad_token_id=self.t_tokenizer.pad_token_id
        )
        generated_ids_trimmed_p = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids_p)
        ]
        descriptions = self.p_processor.batch_decode(
            generated_ids_trimmed_p, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        descriptions_raw = self.p_processor.batch_decode(
            generated_ids_trimmed_p, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        # print(f'\n\n##Perceiver text context: {self.p_processor.batch_decode(generated_ids_p, skip_special_tokens=False)}')

        t_prompt_template = '{question} This is a detailed description of the image: {description}'
        t_texts = []
        for i in range(len(descriptions)):
            print(f'\n\n##Perception out-{i}: {descriptions[i]}')
            # print(f'\n\n##Raw perception out-{i}: {descriptions_raw[i]}')

            prompt = t_prompt_template.format(
                question=questions[i], description=descriptions[i].strip())
            message = [{"role": "user", "content": prompt}]
            t_texts.append(self.t_tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True, enable_thinking=True
            ))
            print(f'\n\n##T-Message-{i}: {t_texts[-1]}')

        model_inputs_t = self.t_tokenizer(
            t_texts, return_tensors="pt", padding=True).to(self.thinker.device)

        # print(
        #     f'Thinker generation config: {self.thinker.generation_config.to_dict()}')
        thinker_generation_params = kwargs.get('thinker_generation_params', {})
        thinker_generation_params['max_new_tokens'] = thinker_generation_params.get(
            'max_new_tokens', 32768)

        generated_ids_t = self.thinker.generate(
            **model_inputs_t,
            **thinker_generation_params
        )

        final_responses = []
        for i in range(len(images)):
            output_ids = generated_ids_t[i][len(
                model_inputs_t.input_ids[i]):].tolist()
            try:
                # 寻找 </think> token (151668)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = self.t_tokenizer.decode(
                output_ids[:index], skip_special_tokens=True).strip("\n")
            print(f"\n\n##Thinking content-{i}: {thinking_content}")

            content = self.t_tokenizer.decode(
                output_ids[index:], skip_special_tokens=True).strip("\n")
            final_responses.append(content)
            print(f"\n\n##Answer content-{i}: {content}")

        return generated_ids_p, generated_ids_t, final_responses

    @can_return_tuple
    @auto_docstring
    def forward(self, *args, **kwargs):
        if self.forward_p:
            return self.perceiver(*args, **kwargs)
        elif self.forward_t:
            return self.thinker(*args, **kwargs)
        else:
            raise ValueError(f'Illegal state where {self.forward_p=} and {self.forward_t=}')

    # def forward