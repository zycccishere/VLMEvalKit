from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen25
import transformers.models.qwen3.modeling_qwen3 as qwen3
from transformers import (Qwen2_5_VLModel, Qwen2Config,
                          Qwen2PreTrainedModel, AutoConfig)
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.masking_utils import (ALL_MASK_ATTENTION_FUNCTIONS,
                                        BlockMask,
                                        _is_torch_greater_or_equal_than_2_6,
                                        and_masks,
                                        causal_mask_function,
                                        create_sliding_window_causal_mask,
                                        or_masks,
                                        packed_sequence_mask_function)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
# from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3Model, eager_attention_forward
# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VLRotaryEmbedding, apply_multimodal_rotary_pos_emb
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring
from transformers.utils.deprecation import deprecate_kwarg
try:
    from transformers.masking_utils import _is_torch_xpu_available
except:
    _is_torch_xpu_available = False
# from transformers.masking_utils import find_packed_sequence_indices


def find_packed_sequence_indices(position_ids: torch.Tensor) -> torch.Tensor:
    """
    Find the indices of the sequence to which each new query token in the sequence belongs when using packed
    tensor format (i.e. several sequences packed in the same batch dimension).

    Args:
        position_ids (`torch.Tensor`)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.

    Returns:
        A 2D tensor where each similar integer indicates that the tokens belong to the same sequence. For example, if we
        pack 3 sequences of 2, 3 and 1 tokens respectively along a single batch dim, this will return [[0, 0, 1, 1, 1, 2]].
    """
    # What separate different sequences is when 2 consecutive positions_ids are separated by more than 1. So
    # taking the diff (by prepending the first value - 1 to keep correct indexing) and applying cumsum to the result
    # gives exactly the sequence indices
    # Note that we assume that a single sequence cannot span several batch dimensions, i.e. 1 single sequence
    # cannot be part of the end of the first batch dim and the start of the 2nd one for example
    # We just need the diff on this first value to be 1
    first_dummy_value = position_ids[:, :1] - 1
    position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
    packed_sequence_mask = (position_diff < 0).cumsum(-1)

    # Here it would be nice to return None if we did not detect packed sequence format, i.e. if `packed_sequence_mask[:, -1] == 0`
    # but it causes issues with export
    return packed_sequence_mask


def _preprocess_mask_arguments(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[Union[torch.Tensor, BlockMask]],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor],
    layer_idx: Optional[int],
) -> tuple[bool, Optional[Union[torch.Tensor, BlockMask]], int, int]:
    # If the mask is already 4D, simply return as-is (it was already prepared, or it is custom)
    if isinstance(attention_mask, (torch.Tensor, BlockMask)) and len(attention_mask.shape) == 4:
        return True, attention_mask, None, None, None

    # For TGI/vLLM backends, or other custom attention without equivalent mask creation: we don't need a mask!
    if config._attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        return True, None, None, None, None

    # Move the mask to correct device, and potentially switch dtype for efficiency
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.to(
            device=cache_position.device, dtype=torch.bool)

    # If using a cache, it can give all information about mask sizes based on seen tokens
    if past_key_values is not None:
        kv_length, kv_offset = past_key_values.get_mask_sizes(
            cache_position, layer_idx)
    # Otherwise, the sizes are simply the input sizes
    else:
        kv_length, kv_offset = input_embeds.shape[1], 0

    # We check the position_ids for potential packed sequence format
    packed_sequence_mask = None
    if position_ids is not None and attention_mask is None and past_key_values is None:
        batch_size = input_embeds.shape[0]
        packed_sequence_mask = find_packed_sequence_indices(position_ids[0])
        print(
            f'packed_sequence_mask: {packed_sequence_mask.shape}', flush=True)

    return False, attention_mask, packed_sequence_mask, kv_length, kv_offset


def create_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor] = None,
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
) -> Optional[Union[torch.Tensor, BlockMask]]:
    # If we have an hybrid cache structure, here we want to create the mask for the full layers
    if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(False)
    else:
        layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx
    )
    if early_exit:
        return attention_mask

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    mask_factory_function = causal_mask_function
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    if _is_torch_xpu_available:
        allow_is_causal_skip = True
    else:
        allow_is_causal_skip = not getattr(
            past_key_values, "is_compileable", False)

    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError(
                "Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(
            mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError(
                "Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(
            mask_factory_function, and_mask_function)
        allow_is_causal_skip = False

    if packed_sequence_mask is not None and _is_torch_greater_or_equal_than_2_6:
        mask_factory_function = and_masks(
            mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False

    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,
        dtype=dtype,
        config=config,
    )
    return causal_mask


class Qwen3Attention(qwen3.Qwen3Attention):
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(
            hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(
            hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            hidden_shape).transpose(1, 2)

        # 获取 3D 的 cos 和 sin，用于多模态 RoPE
        cos, sin = position_embeddings

        # 调用多模态的 RoPE 函数
        mrope_section = self.rope_scaling["mrope_section"]
        query_states, key_states = qwen25.apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, mrope_section
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos,
                            "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = qwen3.eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                assert False, (
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DecoderLayer(qwen3.Qwen3DecoderLayer):
    def __init__(self, config: qwen3.Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)


class Qwen3Model(qwen3.Qwen3PreTrainedModel):
    def __init__(self, config: qwen3.Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx)
             for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = qwen3.Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = qwen3.Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self._skip_layers_logged = False  # 用于标记是否已打印跳过层的日志

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        For transformers library version compatability.
        """
        return self.embed_tokens

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length(
            ) if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
                    **mask_kwargs)

        hidden_states = inputs_embeds
        all_hidden_states = ()

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 处理perceiver跳过层的逻辑
        skip_layers = getattr(self.config, 'perceiver_skip_layers', None)
        skip_start = None
        skip_end = None
        skipped_layers = []

        if skip_layers is not None and isinstance(skip_layers, (list, tuple)) and len(skip_layers) == 2:
            skip_start, skip_end = skip_layers
            # 只在第一次forward时打印配置信息
            if not self._skip_layers_logged:
                print(f"[PERCEIVER SKIP LAYERS] 配置跳过层范围: [{skip_start}, {skip_end}] (包含边界)", flush=True)
                print(f"[PERCEIVER SKIP LAYERS] 总层数: {self.config.num_hidden_layers}, 将跳过层: {list(range(skip_start, skip_end + 1))}", flush=True)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            # 检查是否需要跳过当前层
            if skip_start is not None and skip_end is not None:
                if skip_start <= layer_idx <= skip_end:
                    skipped_layers.append(layer_idx)
                    all_hidden_states += (hidden_states,)
                    continue

            all_hidden_states += (hidden_states,)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        # 打印本次forward实际跳过的层（仅在第一次forward时打印详细信息）
        if not self._skip_layers_logged:
            if skipped_layers:
                print(f"[PERCEIVER SKIP LAYERS] 本次forward实际跳过的层: {skipped_layers}", flush=True)
            else:
                print(f"[PERCEIVER SKIP LAYERS] 本次forward实际跳过的层: [] (未跳过任何层)", flush=True)
            self._skip_layers_logged = True

        hidden_states = self.norm(hidden_states)
        all_hidden_states += (hidden_states,)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,# for transformers library version compatability
            past_key_values=past_key_values if use_cache else None,
        )


class LLaVABaselineConfig(Qwen2Config):
    model_type = "llava_baseline"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self,
                 vit_path='/path/to/home/Qwen2.5-VL-3B-Instruct',
                 llm_path='/path/to/home/Qwen3-0.6B',
                 perceiver_skip_layers=None,
                 **kwargs):
        self.vit_path = vit_path
        self.llm_path = llm_path
        # perceiver_skip_layers: None 或 [start, end] 表示跳过从start到end（包含）的层
        self.perceiver_skip_layers = perceiver_skip_layers
        super().__init__(**kwargs)

        # Remove text_config and vision_config if they exist as dicts
        # to prevent GenerationConfig from trying to call .to_dict() on them
        if hasattr(self, 'text_config') and isinstance(self.text_config, dict):
            delattr(self, 'text_config')
        if hasattr(self, 'vision_config') and isinstance(self.vision_config, dict):
            delattr(self, 'vision_config')


class LLaVABaselinePreTrainedModel(Qwen2PreTrainedModel):
    config_class = LLaVABaselineConfig


class LLaVABaselineModel(LLaVABaselinePreTrainedModel):
    def __init__(self, config: LLaVABaselineConfig):
        super().__init__(config)
        # self.vlm = Qwen2_5_VLModel.from_pretrained(
        #     config.vit_path, low_cpu_mem_usage=True)
        # self.vlm.language_model = Qwen3Model.from_pretrained(config.llm_path)
        vlm_config = AutoConfig.from_pretrained(config.vit_path)
        language_config = AutoConfig.from_pretrained(config.llm_path)
        # 将perceiver_skip_layers配置传递到language_config中
        if hasattr(config, 'perceiver_skip_layers'):
            language_config.perceiver_skip_layers = config.perceiver_skip_layers
        self.vlm = Qwen2_5_VLModel(vlm_config)
        self.vlm.language_model = Qwen3Model(language_config)
        self.vlm.language_model.rotary_emb = qwen25.Qwen2_5_VLRotaryEmbedding(
            config=config)

        # Set rope_scaling for each attention layer
        for layer in self.vlm.language_model.layers:
            layer.self_attn.rope_scaling = self.vlm.config.rope_scaling

        # Adapt patch merger MLP output dimension to match LLM hidden size
        llm_hidden_size = self.vlm.language_model.config.hidden_size
        patch_merger = self.vlm.visual.merger
        mlp_input_dim = patch_merger.hidden_size
        original_output_dim = patch_merger.mlp[2].out_features
        if original_output_dim != llm_hidden_size:
            new_mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, mlp_input_dim),
                nn.GELU(),
                nn.Linear(mlp_input_dim, llm_hidden_size)
            )
            patch_merger.mlp = new_mlp

        self.config: LLaVABaselineConfig

    def forward(self, *args, **kwargs):
        return self.vlm.forward(*args, **kwargs)


class LLaVABaselineModelForConditionalGeneration(LLaVABaselinePreTrainedModel, GenerationMixin):
    def __init__(self, config: LLaVABaselineConfig):
        super().__init__(config)
        self.model = LLaVABaselineModel(config)
        self.lm_head = nn.Linear(self.model.vlm.language_model.config.hidden_size,
                                 self.model.vlm.language_model.config.vocab_size, bias=False)

        # Tie weights: lm_head shares weights with embed_tokens (standard in Qwen3)
        self.lm_head.weight = self.model.vlm.language_model.embed_tokens.weight

        self.post_init()

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        self.lm_head.weight = self.model.vlm.language_model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.vlm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.vlm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @property
    def language_model(self):
        return self.model.vlm.language_model

    @property
    def visual(self):
        return self.model.vlm.visual

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
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
        **kwargs
    ) -> Union[Tuple, qwen25.Qwen2_5_VLCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size)

            rank = dist.get_rank() if dist.is_initialized() else 'N/A'
            num_items = (labels != -100).sum().item()
            loss_sum = loss.item() * num_items

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return qwen25.Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
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
        return self.model.vlm.prepare_inputs_for_generation(input_ids, **kwargs)


__all__ = ["LLaVABaselineModelForConditionalGeneration", "LLaVABaselineConfig"]
