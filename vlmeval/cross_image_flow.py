from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import types

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


@dataclass
class ImageGridMeta:
    image_index: int
    grid_t: int
    grid_h: int
    grid_w: int
    llm_grid_h: int
    llm_grid_w: int
    token_count: int


def get_visual_backbone(model) -> Any:
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    raise AttributeError("Unable to locate Qwen visual backbone on model.")


def build_image_grid_metas(image_grid_thw: torch.Tensor, spatial_merge_size: int) -> list[ImageGridMeta]:
    if image_grid_thw is None:
        return []
    rows = image_grid_thw.detach().cpu().tolist()
    metas: list[ImageGridMeta] = []
    for image_index, (grid_t, grid_h, grid_w) in enumerate(rows):
        llm_grid_h = int(grid_h // spatial_merge_size)
        llm_grid_w = int(grid_w // spatial_merge_size)
        metas.append(
            ImageGridMeta(
                image_index=int(image_index),
                grid_t=int(grid_t),
                grid_h=int(grid_h),
                grid_w=int(grid_w),
                llm_grid_h=llm_grid_h,
                llm_grid_w=llm_grid_w,
                token_count=int(grid_t * llm_grid_h * llm_grid_w),
            )
        )
    return metas


def build_patch_boxes(*, width: int, height: int, grid_h: int, grid_w: int) -> list[dict[str, float]]:
    patch_w = float(width) / float(grid_w)
    patch_h = float(height) / float(grid_h)
    boxes: list[dict[str, float]] = []
    for row in range(grid_h):
        for col in range(grid_w):
            boxes.append(
                {
                    "row": row,
                    "col": col,
                    "x": col * patch_w,
                    "y": row * patch_h,
                    "w": patch_w,
                    "h": patch_h,
                }
            )
    return boxes


def clamp_box_to_image(box: dict[str, float], width: int, height: int) -> dict[str, float]:
    x = max(0.0, min(float(box["x"]), float(width)))
    y = max(0.0, min(float(box["y"]), float(height)))
    w = max(1.0, min(float(box["w"]), float(width) - x))
    h = max(1.0, min(float(box["h"]), float(height) - y))
    return {"x": x, "y": y, "w": w, "h": h}


def box_to_patch_indices(
    *,
    box: dict[str, float],
    patch_boxes: list[dict[str, float]],
) -> list[int]:
    x0 = float(box["x"])
    y0 = float(box["y"])
    x1 = x0 + float(box["w"])
    y1 = y0 + float(box["h"])
    indices: list[int] = []
    for idx, patch in enumerate(patch_boxes):
        cx = float(patch["x"]) + float(patch["w"]) / 2.0
        cy = float(patch["y"]) + float(patch["h"]) / 2.0
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            indices.append(idx)
    return indices


def exact_diagonal_indices(*, image2_patches: list[dict[str, float]], image1_patches: list[dict[str, float]]) -> list[int]:
    if len(image2_patches) != len(image1_patches):
        limit = min(len(image2_patches), len(image1_patches))
        return list(range(limit))
    return list(range(len(image2_patches)))


def compute_viewer_grid_shape(grid_h: int, grid_w: int, max_side: int) -> tuple[int, int]:
    if max(grid_h, grid_w) <= max_side:
        return grid_h, grid_w
    scale = max(float(grid_h) / float(max_side), float(grid_w) / float(max_side))
    viewer_h = max(1, int(round(grid_h / scale)))
    viewer_w = max(1, int(round(grid_w / scale)))
    return viewer_h, viewer_w


def partition_bounds(size: int, parts: int) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    for idx in range(parts):
        start = int(round(size * idx / parts))
        end = int(round(size * (idx + 1) / parts))
        bounds.append((start, max(start + 1, end)))
    return bounds


def downsample_cross_image_map(
    *,
    q_to_k: torch.Tensor,
    query_grid_h: int,
    query_grid_w: int,
    key_grid_h: int,
    key_grid_w: int,
    max_side: int,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    if q_to_k.shape[0] != query_grid_h * query_grid_w:
        raise ValueError("Query grid shape does not match q_to_k row count.")
    if q_to_k.shape[1] != key_grid_h * key_grid_w:
        raise ValueError("Key grid shape does not match q_to_k column count.")

    viewer_q_h, viewer_q_w = compute_viewer_grid_shape(query_grid_h, query_grid_w, max_side)
    viewer_k_h, viewer_k_w = compute_viewer_grid_shape(key_grid_h, key_grid_w, max_side)

    if (viewer_q_h, viewer_q_w) == (query_grid_h, query_grid_w) and (viewer_k_h, viewer_k_w) == (key_grid_h, key_grid_w):
        return q_to_k, (viewer_q_h, viewer_q_w), (viewer_k_h, viewer_k_w)

    tensor = q_to_k.reshape(query_grid_h, query_grid_w, key_grid_h, key_grid_w)
    q_row_bins = partition_bounds(query_grid_h, viewer_q_h)
    q_col_bins = partition_bounds(query_grid_w, viewer_q_w)
    k_row_bins = partition_bounds(key_grid_h, viewer_k_h)
    k_col_bins = partition_bounds(key_grid_w, viewer_k_w)

    output = q_to_k.new_zeros((viewer_q_h, viewer_q_w, viewer_k_h, viewer_k_w))
    for q_row_idx, (q_row_start, q_row_end) in enumerate(q_row_bins):
        for q_col_idx, (q_col_start, q_col_end) in enumerate(q_col_bins):
            query_block = tensor[q_row_start:q_row_end, q_col_start:q_col_end].mean(dim=(0, 1))
            for k_row_idx, (k_row_start, k_row_end) in enumerate(k_row_bins):
                for k_col_idx, (k_col_start, k_col_end) in enumerate(k_col_bins):
                    output[q_row_idx, q_col_idx, k_row_idx, k_col_idx] = query_block[
                        k_row_start:k_row_end, k_col_start:k_col_end
                    ].sum()
    return output.reshape(viewer_q_h * viewer_q_w, viewer_k_h * viewer_k_w), (viewer_q_h, viewer_q_w), (viewer_k_h, viewer_k_w)


def round_tensor(tensor: torch.Tensor, digits: int = 6) -> list[list[float]]:
    scale = float(10 ** digits)
    rounded = torch.round(tensor.detach().cpu().float() * scale) / scale
    return rounded.tolist()


def summarize_cross_image_map(
    *,
    q_to_k: torch.Tensor,
    target_query_indices: list[int],
    target_key_indices: list[int],
    exact_diag_indices: list[int],
) -> dict[str, float]:
    if q_to_k.numel() == 0:
        return {
            "mean_cross_image_mass": float("nan"),
            "mean_target_box_mass_full_query": float("nan"),
            "mean_target_box_mass_target_query": float("nan"),
            "mean_diag_mass_full_query": float("nan"),
            "mean_diag_mass_target_query": float("nan"),
        }

    cross_image_mass = q_to_k.sum(dim=-1)
    normalized = q_to_k / cross_image_mass.unsqueeze(-1).clamp_min(1e-8)

    target_mass = (
        normalized[:, target_key_indices].sum(dim=-1) if target_key_indices else normalized.new_zeros(normalized.shape[0])
    )

    diag_mass_values: list[float] = []
    for q_idx, k_idx in enumerate(exact_diag_indices):
        if q_idx >= normalized.shape[0] or k_idx >= normalized.shape[1]:
            break
        diag_mass_values.append(float(normalized[q_idx, k_idx].item()))
    diag_mass = torch.tensor(diag_mass_values) if diag_mass_values else torch.empty(0)

    def mean_selected(values: torch.Tensor, indices: list[int]) -> float:
        if values.numel() == 0:
            return float("nan")
        if not indices:
            return float("nan")
        valid = [idx for idx in indices if 0 <= idx < values.shape[0]]
        if not valid:
            return float("nan")
        return float(values[valid].mean().item())

    return {
        "mean_cross_image_mass": float(cross_image_mass.mean().item()),
        "mean_target_box_mass_full_query": float(target_mass.mean().item()) if target_mass.numel() else float("nan"),
        "mean_target_box_mass_target_query": mean_selected(target_mass, target_query_indices),
        "mean_diag_mass_full_query": float(diag_mass.mean().item()) if diag_mass.numel() else float("nan"),
        "mean_diag_mass_target_query": mean_selected(diag_mass, target_query_indices),
    }


class CrossImageAttentionTracer:
    def __init__(self, attn_modules: dict[int, Any]):
        self.attn_modules = attn_modules
        self.original_forward: dict[int, Any] = {}
        self.records: list[dict[str, Any]] = []
        self.query_positions: list[int] = []
        self.key_positions: list[int] = []

    def set_trace_positions(self, *, query_positions: list[int], key_positions: list[int]) -> None:
        self.query_positions = [int(x) for x in query_positions]
        self.key_positions = [int(x) for x in key_positions]

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

                query_states = module.q_proj(hidden_states)
                key_states = module.k_proj(hidden_states)
                value_states = module.v_proj(hidden_states)

                query_states = query_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)

                if position_embeddings is None:
                    raise ValueError("position_embeddings is required for traced attention.")
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

                if q_len > 1 and tracer.query_positions and tracer.key_positions:
                    query_positions = [pos for pos in tracer.query_positions if 0 <= pos < q_len]
                    key_positions = [pos for pos in tracer.key_positions if 0 <= pos < key_states_for_scores.shape[-2]]
                    if query_positions and key_positions:
                        query_index = torch.tensor(query_positions, device=query_states.device, dtype=torch.long)
                        key_index = torch.tensor(key_positions, device=query_states.device, dtype=torch.long)
                        selected_queries = query_states.index_select(2, query_index)
                        selected_scores = torch.matmul(
                            selected_queries, key_states_for_scores.transpose(2, 3)
                        ) * module.scaling
                        if attention_mask is not None:
                            causal_mask = attention_mask[:, :, query_index, : key_states_for_scores.shape[-2]]
                            selected_scores = selected_scores + causal_mask
                        selected_attn = F.softmax(selected_scores, dim=-1, dtype=torch.float32)
                        tracer.records.append(
                            {
                                "layer": int(_layer_idx),
                                "query_positions": query_positions,
                                "key_positions": key_positions,
                                "q_to_k": selected_attn.index_select(3, key_index).mean(dim=1)[0].detach().cpu(),
                            }
                        )

                attention_interface = eager_attention_forward
                if module.config._attn_implementation != "eager":
                    attention_interface = ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation]

                attn_output, attn_weights = attention_interface(
                    module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not module.training else module.attention_dropout,
                    scaling=module.scaling,
                    sliding_window=module.sliding_window,
                    **kwargs,
                )

                attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
                attn_output = module.o_proj(attn_output)
                if not output_attentions:
                    attn_weights = None
                return attn_output, attn_weights, past_key_value

            attn_module.forward = types.MethodType(instrumented_forward, attn_module)

    def restore(self) -> None:
        for layer_idx, attn_module in self.attn_modules.items():
            attn_module.forward = self.original_forward[layer_idx]

    def reset(self) -> None:
        self.records.clear()


def mean_records_by_layer(records: list[dict[str, Any]], selected_layers: list[int]) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    per_layer: dict[int, torch.Tensor] = {}
    for record in records:
        per_layer[int(record["layer"])] = record["q_to_k"].float()
    ordered = [per_layer[layer] for layer in selected_layers if layer in per_layer]
    if not ordered:
        raise RuntimeError("No traced attention layers were captured.")
    return per_layer, torch.stack(ordered, dim=0).mean(dim=0)


def image_to_data_uri(path: Path) -> str:
    import base64
    import mimetypes

    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
