"""V2-only no-write for vLLM 0.31 FlexAttention, without a new attention kernel.

Uses the native paged-KV mapping and direct BlockMask builder. Only the
predicate is extended. Worker-local installation is explicit and version
scoped; no site-packages files or default model behavior are changed.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
from dataclasses import dataclass

import numpy as np
import torch

from vlmeval.probes.readout_v2 import contiguous_token_spans


@dataclass(frozen=True)
class Bounds:
    key_start: int
    query_start: int
    query_end: int  # exclusive; includes both vision delimiters

    @property
    def removed_edges(self):
        return (self.query_end-self.query_start)*(self.query_start-self.key_start)


def image_bounds(ids, image_id=151655, start_id=151652, end_id=151653):
    spans = contiguous_token_spans(list(ids), image_id)
    if len(spans) != 2:
        raise ValueError(f'Expected two visual blocks, got {len(spans)}')
    for s in spans:
        if s['start'] < 1 or ids[s['start']-1] != start_id or ids[s['end']+1] != end_id:
            raise ValueError('Image delimiter contract mismatch')
    return Bounds(spans[0]['start']-1, spans[1]['start']-1, spans[1]['end']+2)


def allowed_no_write(q, k, bounds: Bounds):
    """Dense independent-facing specification; preserves causal and system prefix."""
    blocked = ((q >= bounds.query_start) & (q < bounds.query_end)
               & (k >= bounds.key_start) & (k < bounds.query_start))
    return (q >= k) & ~blocked


def paged_predicate(metadata, native_mask, ranges):
    """Map packed queries to requests BEFORE indexing their intervention ranges.

    The native logical_mask_mod's b is the attention batch dimension (usually
    zero), NOT the request index. Using b for per-request ranges is incorrect.
    """
    def mask(b, h, q, physical_k):
        valid, logical_q, logical_k = metadata._convert_physical_to_logical(
            metadata.doc_ids, q, physical_k)
        request = metadata.doc_ids[q]
        lo = ranges[request, 0]
        start = ranges[request, 1]
        end = ranges[request, 2]
        enabled = ranges[request, 3] == 1
        blocked = (enabled & (logical_q >= start) & (logical_q < end)
                   & (logical_k >= lo) & (logical_k < start))
        return valid & native_mask(b, h, q, physical_k) & ~blocked
    return mask


class GenerationAttentionWorker:
    """Native worker_extension_cls; methods called through LLM.collective_rpc."""

    def install_generation_attention(self, token_ids):
        from vllm.v1.attention.backends.flex_attention import (
            FlexAttentionMetadata, FlexAttentionMetadataBuilder)
        version = importlib.metadata.version('vllm')
        if not version.startswith('0.31.'):
            raise RuntimeError(f'This adapter is verified only against vLLM 0.31, got {version}')
        if hasattr(self, '_generation_attention'):
            raise RuntimeError('Intervention already installed')
        runner = self.model_runner
        config = self.vllm_config
        if not config.model_config.enforce_eager or config.cache_config.enable_prefix_caching:
            raise ValueError('Require enforce_eager=True and enable_prefix_caching=False')
        if config.scheduler_config.async_scheduling:
            raise ValueError('Require async_scheduling=False for request-metadata ownership')
        if not config.use_v2_model_runner:
            raise ValueError('Require the native vLLM 0.31 V2 model runner')
        state = dict(mode='native', token_ids=token_ids, audit={}, steps=0,mask_steps=0,mask_active=False,
                     prompt_ids={}, batch=None,
                     ranges=torch.zeros((config.scheduler_config.max_num_seqs,4),
                                        dtype=torch.int32,device=self.device))
        self._generation_attention = state
        native_build = FlexAttentionMetadataBuilder.build
        native_paged = FlexAttentionMetadata.get_paged_mask_mod
        native_add = runner.req_states.add_request
        native_remove = runner.req_states.remove_request
        native_prepare = runner.model_state.prepare_attn

        def add_request(req_id, prompt_len, all_token_ids, *a, **kw):
            state['prompt_ids'][req_id] = list(all_token_ids[:prompt_len])
            return native_add(req_id,prompt_len,all_token_ids,*a,**kw)

        def remove_request(req_id):
            state['prompt_ids'].pop(req_id,None)
            return native_remove(req_id)

        def prepare_attn(input_batch, *a, **kw):
            # V2 sorts scheduled requests before constructing this InputBatch.
            # Persistent request-state slot indices are NOT batch indices.
            state['batch'] = input_batch
            return native_prepare(input_batch,*a,**kw)

        runner.req_states.add_request = add_request
        runner.req_states.remove_request = remove_request
        runner.model_state.prepare_attn = prepare_attn

        def build(builder, common_prefix_len, common_attn_metadata, fast_build=False):
            state['steps'] += 1
            state['mask_active'] = state['mode']=='all_visible'
            if state['mode'] != 'native':
                batch = state['batch']
                n = common_attn_metadata.num_reqs
                if n != len(batch.req_ids):
                    raise RuntimeError('Request sub-batching is not supported by this adapter')
                values = np.zeros((n,4), dtype=np.int32)
                for i, request_id in enumerate(batch.req_ids):
                    key = state['mode']+':'+request_id
                    if key not in state['audit']:
                        ids = state['prompt_ids'][request_id]
                        length = len(ids)
                        bounds = image_bounds(ids, **state['token_ids'])
                        state['audit'][key] = dict(request_id=request_id, mode=state['mode'],
                            prompt_tokens=length, prompt_sha256=hashlib.sha256(
                                np.asarray(ids,dtype=np.int32).tobytes()).hexdigest(),
                            key_start=bounds.key_start,query_start=bounds.query_start,
                            query_end=bounds.query_end,removed_edges=bounds.removed_edges
                                if state['mode']=='no_write' else 0)
                    record = state['audit'][key]
                    offset = int(batch.num_computed_tokens_np[i])
                    count = int(batch.num_scheduled_tokens[i])
                    record['min_query_offset'] = min(record.get('min_query_offset',offset),offset)
                    record['max_query_offset'] = max(record.get('max_query_offset',offset),offset)
                    record['prefill_steps'] = record.get('prefill_steps',0)+int(offset<record['prompt_tokens'])
                    overlap = max(0, min(offset+count, record['query_end'])
                                  - max(offset, record['query_start']))
                    record['v2_query_tokens_seen'] = record.get('v2_query_tokens_seen', 0)+overlap
                    if overlap:
                        state['mask_active'] = True
                    values[i] = [record['key_start'],record['query_start'],record['query_end'],
                                 int(state['mode']=='no_write')]
                if state['mask_active']:
                    state['mask_steps'] += 1
                    state['ranges'][:n].copy_(torch.from_numpy(values).to(self.device))
            result = native_build(builder,common_prefix_len,common_attn_metadata,fast_build)
            if state['mode'] != 'native':
                # Direct builder has no "fully allowed" blocks that bypass the
                # predicate. Check this executable contract on the actual result.
                if not result.direct_build:
                    raise RuntimeError('Require native direct BlockMask builder')
                if result.block_mask is not None and result.block_mask.full_kv_num_blocks is not None:
                    raise RuntimeError('Unexpected fully-allowed sparse blocks')
            return result

        def get_paged(metadata):
            native = native_paged(metadata)
            return native if not state['mask_active'] else paged_predicate(metadata,native,state['ranges'])

        FlexAttentionMetadataBuilder.build = build
        FlexAttentionMetadata.get_paged_mask_mod = get_paged
        return dict(version=version,device=str(self.device),installed=True)

    def set_generation_intervention(self, mode):
        if mode not in ['native','all_visible','no_write']:
            raise ValueError(mode)
        self._generation_attention['mode'] = mode
        self._generation_attention['audit'] = {}
        self._generation_attention['steps'] = 0
        self._generation_attention['mask_steps'] = 0
        return mode

    def generation_intervention_audit(self):
        s = self._generation_attention
        return dict(mode=s['mode'],steps=s['steps'],mask_steps=s['mask_steps'],requests=list(s['audit'].values()))
