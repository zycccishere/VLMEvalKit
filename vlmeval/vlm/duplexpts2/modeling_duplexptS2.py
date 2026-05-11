import torch
from PIL import Image
from typing import Callable, Optional, Union
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen3ForCausalLM
from qwen_vl_utils import process_vision_info
from typing import List, Dict, Union
from transformers import Qwen2PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.processing_utils import Unpack
from transformers.utils import is_torchdynamo_compiling, ModelOutput
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import KwargsForCausalLM, Qwen2_5_VLModelOutputWithPast
from .configuration_duplexptS2 import DuplexPTS2Config
from dataclasses import dataclass


# text, image => ids + features => out_ids
# text, image => ids + features => perceiver out_ids => thinker input_ids => thinker out_ids
# 实现 LLM 类

from transformers.utils import auto_docstring, can_return_tuple
from transformers.cache_utils import Cache, DynamicCache

import os

p_prompt_out = os.getenv('P_PROMPT_OUT', 'Encode the image into {num_feat} tokens, including information related to the question. Here is the question: {question}')

t_prompt_out = os.getenv('T_PROMPT_OUT', '{question} Image: ')

class DuplexPTS2PreTrainedModel(Qwen2PreTrainedModel):
    config_class = DuplexPTS2Config


def add_special_tokens(tkz):
    additional_special_tokens = [f'<im_msg-{i}>' for i in range(128)]
    tkz.add_special_tokens({
        'additional_special_tokens': additional_special_tokens
    })
    mapping = {
        tok: tkz._convert_token_to_id_with_added_voc(tok) for tok in additional_special_tokens
    }
    return tkz, mapping


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava outputs, with hidden states and attentions.
    """
)
class DuplexPTS2OutputWithPast(ModelOutput):
    r"""
    past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None


class DuplexPTS2Model(DuplexPTS2PreTrainedModel, GenerationMixin):
    def __init__(self, config: DuplexPTS2Config):
        super().__init__(config)

        self.perceiver = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.perceiver_name_or_path,
        )
        self.p_processor = AutoProcessor.from_pretrained(
            self.config.p_processor_name_or_path)
        self.p_processor.tokenizer.padding_side = "left"

        self.thinker = Qwen3ForCausalLM.from_pretrained(
            self.config.thinker_name_or_path,
        )
        self.t_tokenizer = AutoTokenizer.from_pretrained(
            self.config.t_tokenizer_name_or_path, padding_side="left")
        # print(self.t_tokenizer)

        self.linear_align_dim = torch.nn.Linear(
            self.perceiver.config.hidden_size, self.thinker.config.hidden_size)
        # print(f'Linear alinger: {self.linear_align_dim}')

        self.config: DuplexPTS2Config

        self.prompt_added_list = ["Please select the correct answer from the options above.", '请直接回答选项字母。', 'Please select the correct answer from the options above.', ' Please answer yes or no.', 'Please try to answer the question with short words or phrases if possible.']

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

    def get_visual_message_tokens(self):
        size = self.config.visual_bandwidth
        tokens = [f'<im_msg-{i}>' for i in range(size)]
        return tokens

    def get_visual_message_token_ids(self, model):
        tokens = self.get_visual_message_tokens()
        if model == 'p':
            ids = self.p_processor.tokenizer.convert_tokens_to_ids(tokens)
        elif model == 't':
            ids = self.t_tokenizer.convert_tokens_to_ids(tokens)
        else:
            raise NotImplementedError
        return ids

    def get_visual_message(self):
        message = ''.join(self.get_visual_message_tokens())
        return message

    def chat(self, images, msgs, *args, **kwargs):
        assert len(images) == len(msgs)
        assert args == ()
        assert 'max_new_tokens' not in kwargs

        p_prompt_template = p_prompt_out
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
            processed_image = self._resize_image_if_needed(pil_image)

            p_images.append(processed_image)


            question = msg_list[0]['content']
            questions.append(question)

            # for p_message, remove the added prompt
            if "{question}" in p_prompt_template:
                p_prompt_template_final = p_prompt_template.format(
                    num_feat=self.config.visual_bandwidth, question=question)
            else:
                p_prompt_template_final = p_prompt_template.format(
                    num_feat=self.config.visual_bandwidth)
            for added_prompt in self.prompt_added_list:
                if added_prompt in p_prompt_template_final:
                    p_prompt_template_final = p_prompt_template_final.replace(
                        added_prompt, '')
            p_prompt_template_final = p_prompt_template_final.strip()

            p_message = [
                {'role': 'user', 'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': p_prompt_template_final}
                ]},
                {'role': 'assisstant', 'content': [
                    {'type': 'text', 'text': self.get_visual_message()}
                ]}
            ]
            print(f'P-Message-{i}: {p_message}', flush=True)
            p_texts.append(self.p_processor.apply_chat_template(
                p_message, tokenize=False, add_generation_prompt=False))

        print(f'{p_texts=}', flush=True)
        perceiver_inputs = self.p_processor(
            text=p_texts,
            images=p_images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # print('Token IDs of perceiver inputs',
            #   perceiver_inputs['input_ids'].tolist())
        # print('Tokens of perceiver inputs', [
            #   self.p_processor.tokenizer.convert_ids_to_tokens(ids) for ids in perceiver_inputs['input_ids']])

        t_prompt_template = t_prompt_out + self.get_visual_message()
        t_texts = []
        for i in range(len(questions)):
            prompt = t_prompt_template.format(question=questions[i])
            message = [{"role": "user", "content": prompt}]
            t_texts.append(self.t_tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True, enable_thinking=True
            ))
            print(f'\n\n##T-Message-{i}: {t_texts[-1]}', flush=True)

        model_inputs_t = self.t_tokenizer(
            t_texts, return_tensors="pt", padding=True).to(self.thinker.device)

        model_inputs_t['input_ids_of_perceiver'] = perceiver_inputs['input_ids']
        model_inputs_t['attention_mask_of_perceiver'] = perceiver_inputs['attention_mask']
        model_inputs_t['pixel_values'] = perceiver_inputs['pixel_values']
        model_inputs_t['image_grid_thw'] = perceiver_inputs['image_grid_thw']

        # print(
        #     f'Thinker generation config: {self.thinker.generation_config.to_dict()}')
        thinker_generation_params = kwargs.get('thinker_generation_params', {})
        thinker_generation_params['max_new_tokens'] = thinker_generation_params.get(
            'max_new_tokens', 32768)

        assert model_inputs_t['pixel_values'] is not None

        with torch.inference_mode():
            generated_ids_t = self.generate(
                **model_inputs_t,
                **thinker_generation_params
            )
        print(f'Thinker output ids: {generated_ids_t}')
        print(
            f'Thinker output toks: {[self.t_tokenizer.convert_ids_to_tokens(ids) for ids in generated_ids_t]}')

        final_responses = []
        for i in range(len(msgs)):
            output_ids = generated_ids_t[i][len(
                model_inputs_t.input_ids[i]):].tolist()
            try:
                # 寻找 </think> token (151668)
                index = len(output_ids) - output_ids[::-1].index(151668)
                print(
                    f'len output_ids: {len(output_ids)}, subtract {output_ids[::-1].index(151668)}')
            except ValueError:
                index = 0

            thinking_content = self.t_tokenizer.decode(
                output_ids[:index], skip_special_tokens=False).strip("\n")
            print(f"\n\n##Thinking content-{i}: {thinking_content}")

            content = self.t_tokenizer.decode(
                output_ids[index:], skip_special_tokens=False).strip("\n")
            final_responses.append(content)
            print(f"\n\n##Answer content-{i}: {content}", flush=True)

        # return [x[0] for x in self.generate([image], [msgs], *args, **kwargs)]
        return final_responses

    # NOTE: All inputs should be considered as inputs to thinker
    #       The thinker consumes multimodal data by calling perceiver
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        input_ids_of_perceiver=None,
        attention_mask_of_perceiver=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        assert pixel_values is not None
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            attention_mask=attention_mask,
            input_ids_of_perceiver=input_ids_of_perceiver,
            attention_mask_of_perceiver=attention_mask_of_perceiver,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )
        # print(f'\n@@@@ prepare inputs for generation', f'##${model_inputs["pixel_values"].shape}$##', flush=True)

        # # Qwen2-5-VL position_ids are prepareed with rope_deltas in forward
        # model_inputs["position_ids"] = None

        assert model_inputs["pixel_values"] is not None
        if cache_position[0] != 0:
            # print(f'Cache hit, skip pixel values encoding', flush=True)
            model_inputs["pixel_values"] = None
            # model_inputs["pixel_values_videos"] = None

        return model_inputs

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids_of_perceiver: torch.LongTensor = None,
        attention_mask_of_perceiver: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:

        t_input_ids = input_ids
        del input_ids

        # START get perceiver message features
        if inputs_embeds is None:
            inputs_embeds = self.thinker.get_input_embeddings()(t_input_ids)

            if pixel_values is not None:
                p_msg_token_ids = self.get_visual_message_token_ids('p')
                p_msg_st_id, p_msg_ed_id = p_msg_token_ids[0], p_msg_token_ids[-1]
                p_msg_st_list = []
                p_msg_ed_list = []

                for perceiver_sample_input_ids in input_ids_of_perceiver:
                    st_indices = (perceiver_sample_input_ids ==
                                  p_msg_st_id).nonzero(as_tuple=True)[0]
                    ed_indices = (perceiver_sample_input_ids ==
                                  p_msg_ed_id).nonzero(as_tuple=True)[0]
                    assert len(st_indices) == 1, f'{st_indices}'
                    assert len(ed_indices) == 1, f'{ed_indices}'

                    p_msg_st_list.append(st_indices[0])
                    p_msg_ed_list.append(ed_indices[0])

                # prevent usage in thinker (causing errors due to different vocab)
                del p_msg_st_id, p_msg_ed_id, p_msg_token_ids

                # print(f'type(self.perceiver) is {type(self.perceiver)}', flush=True)
                out = self.perceiver(input_ids=input_ids_of_perceiver,
                                     pixel_values=pixel_values,
                                     attention_mask=attention_mask_of_perceiver,
                                     image_grid_thw=image_grid_thw,
                                     output_hidden_states=True)

                last_layer_hiddens = out.hidden_states[-1]

                batch_msg = []
                for hiddens, st, ed in zip(last_layer_hiddens, p_msg_st_list, p_msg_ed_list):

                    # Q, H
                    msg_feat = hiddens[st:ed + 1, :]
                    batch_msg.append(msg_feat)

                # (N_img x Q) x H
                image_features = torch.cat(batch_msg, dim=0)
                image_features = self.linear_align_dim(image_features)

                t_msg_token_ids = self.get_visual_message_token_ids('t')
                t_msg_st_id, t_msg_ed_id = t_msg_token_ids[0], t_msg_token_ids[-1]
                n_msg_features = image_features.shape[0]
                msg_mask = (t_input_ids >= t_msg_st_id) & (
                    t_input_ids <= t_msg_ed_id)
                # print(f'LEQ than st: {(t_input_ids >= t_msg_st_id)}')
                # print(f'SEQ than ed: {(t_input_ids <= t_msg_ed_id)}')
                n_msg_tokens = msg_mask.sum()
                if not is_torchdynamo_compiling() and n_msg_tokens != n_msg_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_msg_tokens}, features {n_msg_features}"
                    )

                mask_unsqueezed = msg_mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)

                image_mask = mask_expanded.to(inputs_embeds.device)
                image_features = image_features.to(
                    inputs_embeds.device, inputs_embeds.dtype)

                print(
                    f'inputs_embeds.shape: {inputs_embeds.shape} image_features.shape: {image_features.shape}')
                inputs_embeds = inputs_embeds.masked_scatter(
                    image_mask, image_features)
        # END get perceiver message features

        outputs = self.thinker(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = DuplexPTS2OutputWithPast(
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            logits=outputs.logits
        )
        return output if return_dict else output.to_tuple()
