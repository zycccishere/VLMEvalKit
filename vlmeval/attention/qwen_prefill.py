from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
import types

import numpy as np
import torch
import torch.nn.functional as F

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import ALL_ATTENTION_FUNCTIONS
except Exception:  # pragma: no cover
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


@dataclass(frozen=True)
class PositionGroup:
    name: str
    positions: tuple[int, ...]
    kind: str = ""

    @classmethod
    def from_positions(cls, name: str, positions: Iterable[int], *, kind: str = "") -> "PositionGroup":
        return cls(name=name, positions=tuple(int(pos) for pos in positions), kind=kind)

    @property
    def length(self) -> int:
        return len(self.positions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "positions": [int(pos) for pos in self.positions],
            "length": int(self.length),
        }


@dataclass(frozen=True)
class AttentionMatrixSpec:
    name: str
    query_group: str
    key_group: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "query_group": self.query_group,
            "key_group": self.key_group,
        }


@dataclass(frozen=True)
class AttentionFullMapSpec:
    name: str
    query_groups: tuple[str, ...]
    key_groups: tuple[str, ...]
    heads: tuple[int, ...]

    @classmethod
    def from_groups(
        cls,
        name: str,
        query_groups: Sequence[str],
        key_groups: Sequence[str],
        *,
        heads: Sequence[int],
    ) -> "AttentionFullMapSpec":
        return cls(
            name=str(name),
            query_groups=tuple(str(group) for group in query_groups),
            key_groups=tuple(str(group) for group in key_groups),
            heads=tuple(int(head) for head in heads),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "query_groups": list(self.query_groups),
            "key_groups": list(self.key_groups),
            "heads": [int(head) for head in self.heads],
        }


def get_language_model_layers(model: Any) -> Sequence[Any]:
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    if hasattr(model, "language_model"):
        return model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Unable to locate language-model layers on Qwen model.")


def _dedup_positions(positions: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    deduped: list[int] = []
    for pos in positions:
        pos_int = int(pos)
        if pos_int in seen:
            continue
        seen.add(pos_int)
        deduped.append(pos_int)
    return deduped


def _valid_positions(positions: Iterable[int], limit: int) -> list[int]:
    return [int(pos) for pos in _dedup_positions(positions) if 0 <= int(pos) < limit]


def _coerce_groups(groups: Mapping[str, Iterable[int]] | Sequence[PositionGroup]) -> dict[str, PositionGroup]:
    if isinstance(groups, Mapping):
        return {
            str(name): PositionGroup.from_positions(str(name), positions)
            for name, positions in groups.items()
        }
    return {group.name: group for group in groups}


def _project_qkv(module: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, q_len, _ = hidden_states.size()
    query_states = module.q_proj(hidden_states).view(bsz, q_len, -1, module.head_dim)
    key_states = module.k_proj(hidden_states).view(bsz, q_len, -1, module.head_dim)
    value_states = module.v_proj(hidden_states).view(bsz, q_len, -1, module.head_dim)

    if hasattr(module, "q_norm"):
        query_states = module.q_norm(query_states)
    if hasattr(module, "k_norm"):
        key_states = module.k_norm(key_states)

    return (
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
    )


def _attention_interface(module: Any):
    implementation = getattr(module.config, "_attn_implementation", "eager")
    if implementation == "eager":
        return eager_attention_forward
    return ALL_ATTENTION_FUNCTIONS[implementation]


def _select_attention_mask(
    *,
    attention_mask: torch.Tensor | None,
    query_index: torch.Tensor,
    kv_len: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.dim() == 4:
        sliced = attention_mask[:, :, :, :kv_len]
        return sliced.index_select(2, query_index)
    if attention_mask.dim() == 2:
        return attention_mask[:, None, None, :kv_len]
    raise ValueError(f"Unsupported attention_mask rank for tracing: {attention_mask.dim()}")


class QwenPrefillAttentionTracer:
    """Patch Qwen language self-attention modules and capture prefill group attention.

    The tracer records only selected query rows during q_len > 1 prefill calls. It
    forwards through the model's normal attention implementation after recording,
    so the traced pass remains behaviorally identical to the untraced pass.
    """

    def __init__(
        self,
        attn_modules: Mapping[int, Any],
        *,
        mass_query_groups: Sequence[str],
        mass_key_groups: Sequence[str],
        matrix_specs: Sequence[AttentionMatrixSpec] | None = None,
        full_map_specs: Sequence[AttentionFullMapSpec] | None = None,
        max_full_matrix_elements: int = 4_000_000,
    ) -> None:
        self.attn_modules = dict(attn_modules)
        self.mass_query_groups = [str(name) for name in mass_query_groups]
        self.mass_key_groups = [str(name) for name in mass_key_groups]
        self.matrix_specs = list(matrix_specs or [])
        self.full_map_specs = list(full_map_specs or [])
        self.max_full_matrix_elements = int(max_full_matrix_elements)
        self.original_forward: dict[int, Any] = {}
        self.records: list[dict[str, Any]] = []
        self.sample_id = ""
        self.groups: dict[str, PositionGroup] = {}

    def configure_sample(
        self,
        *,
        sample_id: str,
        groups: Mapping[str, Iterable[int]] | Sequence[PositionGroup],
        matrix_specs: Sequence[AttentionMatrixSpec] | None = None,
        full_map_specs: Sequence[AttentionFullMapSpec] | None = None,
    ) -> None:
        self.sample_id = str(sample_id)
        self.groups = _coerce_groups(groups)
        if matrix_specs is not None:
            self.matrix_specs = list(matrix_specs)
        if full_map_specs is not None:
            self.full_map_specs = list(full_map_specs)

    def patch(self) -> None:
        tracer = self
        for layer_idx, attn_module in self.attn_modules.items():
            self.original_forward[layer_idx] = attn_module.forward

            def instrumented_forward(
                module,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                position_ids: torch.LongTensor | None = None,
                past_key_value=None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: torch.LongTensor | None = None,
                position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
                _layer_idx: int = layer_idx,
                **kwargs,
            ):
                bsz, q_len, _ = hidden_states.size()
                query_states, key_states, value_states = _project_qkv(module, hidden_states)

                if position_embeddings is None:
                    raise ValueError("position_embeddings is required for Qwen prefill attention tracing.")
                cos, sin = position_embeddings
                query_states, key_states = apply_multimodal_rotary_pos_emb(
                    query_states,
                    key_states,
                    cos,
                    sin,
                    module.rope_scaling["mrope_section"],
                )

                if past_key_value is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, module.layer_idx, cache_kwargs
                    )

                key_states_for_scores = repeat_kv(key_states, module.num_key_value_groups)
                tracer._maybe_record_prefill(
                    layer_idx=int(_layer_idx),
                    bsz=int(bsz),
                    q_len=int(q_len),
                    query_states=query_states,
                    key_states_for_scores=key_states_for_scores,
                    attention_mask=attention_mask,
                    scaling=float(module.scaling),
                )

                attention_interface = _attention_interface(module)
                attn_output, attn_weights = attention_interface(
                    module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not module.training else module.attention_dropout,
                    scaling=module.scaling,
                    sliding_window=getattr(module, "sliding_window", None),
                    **kwargs,
                )

                attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
                attn_output = module.o_proj(attn_output)
                if not output_attentions:
                    attn_weights = None
                return attn_output, attn_weights

            attn_module.forward = types.MethodType(instrumented_forward, attn_module)

    def restore(self) -> None:
        for layer_idx, attn_module in self.attn_modules.items():
            if layer_idx in self.original_forward:
                attn_module.forward = self.original_forward[layer_idx]

    def reset(self) -> None:
        self.records.clear()

    def _query_union(self, q_len: int) -> list[int]:
        union: list[int] = []
        query_group_names = list(self.mass_query_groups)
        query_group_names.extend(spec.query_group for spec in self.matrix_specs)
        for spec in self.full_map_specs:
            query_group_names.extend(spec.query_groups)
        for name in query_group_names:
            group = self.groups.get(name)
            if group is None:
                continue
            union.extend(group.positions)
        return _valid_positions(union, q_len)

    def _group_positions(self, name: str, limit: int) -> list[int]:
        group = self.groups.get(name)
        if group is None:
            return []
        return _valid_positions(group.positions, limit)

    def _concat_group_positions(
        self,
        group_names: Sequence[str],
        limit: int,
    ) -> tuple[list[int], list[dict[str, Any]]]:
        positions: list[int] = []
        segments: list[dict[str, Any]] = []
        cursor = 0
        for group_name in group_names:
            group_positions = self._group_positions(group_name, limit)
            positions.extend(group_positions)
            next_cursor = cursor + len(group_positions)
            segments.append(
                {
                    "group": str(group_name),
                    "start": int(cursor),
                    "end": int(next_cursor),
                    "length": int(len(group_positions)),
                }
            )
            cursor = next_cursor
        return positions, segments

    def _maybe_record_prefill(
        self,
        *,
        layer_idx: int,
        bsz: int,
        q_len: int,
        query_states: torch.Tensor,
        key_states_for_scores: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
    ) -> None:
        if q_len <= 1 or not self.groups:
            return
        if bsz != 1:
            raise ValueError("QwenPrefillAttentionTracer currently expects batch size 1.")

        kv_len = int(key_states_for_scores.shape[-2])
        query_positions = self._query_union(q_len)
        if not query_positions:
            return

        query_index = torch.as_tensor(query_positions, device=query_states.device, dtype=torch.long)
        selected_queries = query_states.index_select(2, query_index)
        scores = torch.matmul(selected_queries, key_states_for_scores.transpose(2, 3)) * scaling
        selected_mask = _select_attention_mask(
            attention_mask=attention_mask,
            query_index=query_index,
            kv_len=kv_len,
        )
        if selected_mask is not None:
            scores = scores + selected_mask
        selected_attn = F.softmax(scores, dim=-1, dtype=torch.float32)

        query_rel = {int(pos): idx for idx, pos in enumerate(query_positions)}
        group_masses = []
        for query_group in self.mass_query_groups:
            q_positions = self._group_positions(query_group, q_len)
            q_rel = [query_rel[pos] for pos in q_positions if pos in query_rel]
            if not q_rel:
                continue
            q_rel_tensor = torch.as_tensor(q_rel, device=selected_attn.device, dtype=torch.long)
            attn_q = selected_attn.index_select(2, q_rel_tensor)
            for key_group in self.mass_key_groups:
                key_positions = self._group_positions(key_group, kv_len)
                if not key_positions:
                    continue
                key_tensor = torch.as_tensor(key_positions, device=selected_attn.device, dtype=torch.long)
                mass_per_query = attn_q.index_select(3, key_tensor).sum(dim=-1).squeeze(0)
                mass_by_head = mass_per_query.mean(dim=-1)
                std_by_head = (
                    mass_per_query.std(dim=-1, unbiased=False)
                    if mass_per_query.shape[-1] > 1
                    else torch.zeros_like(mass_by_head)
                )
                group_masses.append(
                    {
                        "query_group": query_group,
                        "key_group": key_group,
                        "query_count": int(len(q_rel)),
                        "key_count": int(len(key_positions)),
                        "mass_by_head": mass_by_head.detach().cpu().float().numpy(),
                        "std_by_head": std_by_head.detach().cpu().float().numpy(),
                    }
                )

        matrices = {}
        for spec in self.matrix_specs:
            q_positions = self._group_positions(spec.query_group, q_len)
            q_rel = [query_rel[pos] for pos in q_positions if pos in query_rel]
            key_positions = self._group_positions(spec.key_group, kv_len)
            if not q_rel or not key_positions:
                continue
            q_rel_tensor = torch.as_tensor(q_rel, device=selected_attn.device, dtype=torch.long)
            key_tensor = torch.as_tensor(key_positions, device=selected_attn.device, dtype=torch.long)
            matrix = (
                selected_attn.index_select(2, q_rel_tensor)
                .index_select(3, key_tensor)
                .squeeze(0)
                .detach()
                .cpu()
                .float()
            )
            query_mean = matrix.mean(dim=1)
            matrix_elements = int(matrix.numel())
            payload = {
                "query_group": spec.query_group,
                "key_group": spec.key_group,
                "query_positions": [int(pos) for pos in q_positions],
                "key_positions": [int(pos) for pos in key_positions],
                "query_mean": query_mean.numpy().astype(np.float16),
                "head_mass": query_mean.sum(dim=-1).numpy().astype(np.float32),
                "matrix_shape": [int(dim) for dim in matrix.shape],
                "matrix_stored": bool(
                    self.max_full_matrix_elements > 0
                    and matrix_elements <= self.max_full_matrix_elements
                ),
            }
            if payload["matrix_stored"]:
                payload["matrix"] = matrix.numpy().astype(np.float16)
            matrices[spec.name] = payload

        full_maps = {}
        for spec in self.full_map_specs:
            q_positions, q_segments = self._concat_group_positions(spec.query_groups, q_len)
            key_positions, key_segments = self._concat_group_positions(spec.key_groups, kv_len)
            q_rel = [query_rel[pos] for pos in q_positions if pos in query_rel]
            if not q_rel or not key_positions:
                continue
            q_rel_tensor = torch.as_tensor(q_rel, device=selected_attn.device, dtype=torch.long)
            key_tensor = torch.as_tensor(key_positions, device=selected_attn.device, dtype=torch.long)
            matrix = (
                selected_attn.index_select(2, q_rel_tensor)
                .index_select(3, key_tensor)
                .squeeze(0)
                .detach()
                .cpu()
                .float()
            )
            head_count = int(matrix.shape[0])
            heads = [int(head) for head in spec.heads if 0 <= int(head) < head_count]
            if not heads:
                heads = list(range(head_count))
            head_tensor = torch.as_tensor(heads, dtype=torch.long)
            matrix = matrix.index_select(0, head_tensor)
            matrix_elements = int(matrix.numel())
            payload = {
                "query_groups": list(spec.query_groups),
                "key_groups": list(spec.key_groups),
                "query_positions": [int(pos) for pos in q_positions],
                "key_positions": [int(pos) for pos in key_positions],
                "query_segments": q_segments,
                "key_segments": key_segments,
                "heads": heads,
                "matrix_shape": [int(dim) for dim in matrix.shape],
                "matrix_stored": bool(
                    self.max_full_matrix_elements > 0
                    and matrix_elements <= self.max_full_matrix_elements
                ),
            }
            if payload["matrix_stored"]:
                payload["matrix"] = matrix.numpy().astype(np.float16)
            full_maps[spec.name] = payload

        self.records.append(
            {
                "sample_id": self.sample_id,
                "layer": int(layer_idx),
                "q_len": int(q_len),
                "kv_len": int(kv_len),
                "query_positions": [int(pos) for pos in query_positions],
                "group_masses": group_masses,
                "matrices": matrices,
                "full_maps": full_maps,
            }
        )
