from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .replay_visual_token_shift import roll_visual_token_blocks


VISUAL_SEQUENCE_ROLL_RIGHT_1 = "visual_sequence_roll_right_1"
VALIDATED_ATTENTION_RETURN_ARITY = {
    "4.53.3": 3,
    "5.5.0": 2,
}


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_raw_bytes(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().reshape(-1)


def qwen_attention_return_arity(version: str) -> int:
    normalized = str(version).strip()
    try:
        return VALIDATED_ATTENTION_RETURN_ARITY[normalized]
    except KeyError as exc:
        raise RuntimeError(
            "Unvalidated Transformers runtime for Qwen attention tracing: "
            f"version={normalized!r}; validated={sorted(VALIDATED_ATTENTION_RETURN_ARITY)}"
        ) from exc


def processed_image_pair_contract(
    model_inputs: dict[str, Any],
    *,
    spatial_merge_size: int,
) -> dict[str, Any]:
    pixel_values = model_inputs.get("pixel_values")
    grid_thw = model_inputs.get("image_grid_thw")
    if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 2:
        raise TypeError(
            "Qwen repeated-image contract requires flattened pixel_values [patch_rows, features], "
            f"got {type(pixel_values)} shape={getattr(pixel_values, 'shape', None)}"
        )
    integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
    if (
        not isinstance(grid_thw, torch.Tensor)
        or grid_thw.ndim != 2
        or grid_thw.shape[1] != 3
        or grid_thw.dtype not in integer_dtypes
    ):
        raise TypeError(
            "Qwen repeated-image contract requires integer image_grid_thw [images, 3], "
            f"got {type(grid_thw)} shape={getattr(grid_thw, 'shape', None)} "
            f"dtype={getattr(grid_thw, 'dtype', None)}"
        )
    merge_size = int(spatial_merge_size)
    if merge_size <= 0:
        raise ValueError(f"invalid Qwen spatial_merge_size={merge_size}")
    grid_rows = [[int(value) for value in row] for row in grid_thw.detach().cpu().tolist()]
    if any(value <= 0 for row in grid_rows for value in row):
        raise ValueError(f"Qwen image_grid_thw entries must be positive, got {grid_rows}")
    patch_row_counts = [int(t * h * w) for t, h, w in grid_rows]
    merge_area = merge_size * merge_size
    if any(count % merge_area != 0 for count in patch_row_counts):
        raise ValueError(
            "Qwen image patch rows must be divisible by spatial_merge_size^2: "
            f"counts={patch_row_counts} spatial_merge_size={merge_size}"
        )
    if len(patch_row_counts) != 2 or sum(patch_row_counts) != int(pixel_values.shape[0]):
        raise ValueError(
            "Qwen IQI repeated-image patch-row split mismatch: "
            f"grid_thw={grid_rows} patch_row_counts={patch_row_counts} "
            f"pixel_values_shape={list(pixel_values.shape)}"
        )
    image1_patch_rows, image2_patch_rows = pixel_values.split(patch_row_counts, dim=0)
    same_grid = grid_rows[0] == grid_rows[1]
    same_shape = image1_patch_rows.shape == image2_patch_rows.shape
    patch_rows_exact = same_shape and torch.equal(image1_patch_rows, image2_patch_rows)
    if same_shape:
        difference = (image1_patch_rows.float() - image2_patch_rows.float()).abs()
        max_abs_diff = float(difference.max().item())
        mean_abs_diff = float(difference.mean().item())
    else:
        max_abs_diff = float("inf")
        mean_abs_diff = float("inf")
    return {
        "validated": bool(same_grid and same_shape and patch_rows_exact),
        "stage": "qwen_processor_output_pre_vit",
        "image_count": 2,
        "grid_thw": grid_rows,
        "spatial_merge_size": merge_size,
        "patch_row_counts": patch_row_counts,
        "pixel_values_shape": list(pixel_values.shape),
        "image_patch_row_shapes": [list(image1_patch_rows.shape), list(image2_patch_rows.shape)],
        "same_grid": bool(same_grid),
        "same_shape": bool(same_shape),
        "patch_rows_exact": bool(patch_rows_exact),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "image1_sha256": tensor_sha256(image1_patch_rows),
        "image2_sha256": tensor_sha256(image2_patch_rows),
    }


class QwenI2VisualSequenceRoll:
    """Capture or roll I2 after Qwen's merger and verify its LLM injection.

    This controller is intentionally scoped to the two-image IQI attention-flow
    probe. The production IQIQ controller keeps its stricter prompt-topology
    contract in ``replay_visual_token_shift.py``.
    """

    def __init__(self, model: torch.nn.Module, *, dump_dir: Path, raw_dump_limit: int = 0) -> None:
        root = getattr(model, "model", model)
        visual = getattr(root, "visual", None)
        language_model = getattr(root, "language_model", None)
        if visual is None:
            raise AttributeError("cannot locate Qwen visual module for sequence-roll probe")
        if language_model is None:
            raise AttributeError("cannot locate Qwen language model for sequence-roll injection validation")
        self.visual = visual
        self.language_model = language_model
        self.dump_dir = Path(dump_dir)
        self.raw_dump_limit = max(0, int(raw_dump_limit))
        self.dump_count = 0
        self._active: dict[str, Any] | None = None
        self._visual_hook_handle = self.visual.register_forward_hook(self._visual_hook, with_kwargs=True)
        self._language_hook_handle = self.language_model.register_forward_pre_hook(
            self._language_pre_hook,
            with_kwargs=True,
        )

    @contextlib.contextmanager
    def sample(
        self,
        *,
        case_id: str,
        intervention: str,
        enabled: bool,
        roll_enabled: bool,
        image1_positions: list[int],
        image2_positions: list[int],
    ) -> Iterator[dict[str, Any]]:
        if self._active is not None:
            raise RuntimeError("Qwen sequence-roll sample contexts cannot be nested")
        state: dict[str, Any] = {
            "applied": False,
            "apply_count": 0,
            "visual_hook_count": 0,
            "language_injection_hook_count": 0,
            "case_id": str(case_id),
            "intervention": str(intervention),
            "capture_enabled": bool(enabled),
            "roll_enabled": bool(roll_enabled),
        }
        if not enabled:
            yield state
            return
        if not image1_positions or not image2_positions:
            raise ValueError("Qwen sequence-roll runtime capture requires non-empty I1 and I2 token spans")
        state["_image1_positions"] = list(map(int, image1_positions))
        state["_image2_positions"] = list(map(int, image2_positions))
        self._active = state
        succeeded = False
        try:
            yield state
            succeeded = True
        finally:
            self._active = None
            if succeeded:
                expected_apply_count = 1 if roll_enabled else 0
                if state["apply_count"] != expected_apply_count:
                    raise RuntimeError(
                        "Qwen I2 sequence-roll apply-count mismatch: "
                        f"expected={expected_apply_count} observed={state['apply_count']}"
                    )
                if state["visual_hook_count"] != 1 or state["language_injection_hook_count"] != 1:
                    raise RuntimeError(
                        "Qwen sequence-roll runtime hooks must each fire exactly once: "
                        f"visual={state['visual_hook_count']} language={state['language_injection_hook_count']}"
                    )
                self._finalize_state(state)

    def _visual_hook(self, module, args, kwargs, output):
        state = self._active
        if state is None:
            return output
        if state["visual_hook_count"] != 0:
            raise RuntimeError("Qwen visual hook fired more than once for one flow-probe sample")

        if isinstance(output, torch.Tensor):
            merged_output = output
            output_kind = "tensor"
        else:
            merged_output = getattr(output, "pooler_output", None)
            output_kind = "base_model_output_with_pooling"
        if not isinstance(merged_output, torch.Tensor) or merged_output.ndim != 2:
            raise TypeError(
                "expected Qwen post-merger visual output [tokens, hidden], "
                f"got output={type(output)} pooler_output={type(merged_output)} "
                f"shape={getattr(merged_output, 'shape', None)}"
            )

        grid_thw = kwargs.get("grid_thw")
        if grid_thw is None and len(args) >= 2:
            grid_thw = args[1]
        if grid_thw is None:
            raise ValueError("Qwen sequence-roll hook did not receive grid_thw")
        grid_rows = grid_thw.detach().cpu().tolist() if hasattr(grid_thw, "detach") else list(grid_thw)
        visual_config = getattr(module, "config", None)
        merge_size = int(
            getattr(module, "spatial_merge_size", 0)
            or getattr(visual_config, "spatial_merge_size", 0)
        )
        if merge_size <= 0:
            raise ValueError(f"invalid Qwen spatial_merge_size={merge_size}")
        token_counts = [int(t * h * w // (merge_size * merge_size)) for t, h, w in grid_rows]
        if sum(token_counts) != int(merged_output.shape[0]):
            raise ValueError(
                f"Qwen visual split mismatch: token_counts={token_counts}, output={list(merged_output.shape)}"
            )
        if len(token_counts) != 2 or token_counts[0] != token_counts[1] or grid_rows[0] != grid_rows[1]:
            raise ValueError(
                "Qwen IQI sequence roll requires exactly two identical image grids, "
                f"got token_counts={token_counts}, grid_thw={grid_rows}"
            )

        chunks = list(merged_output.split(token_counts, dim=0))
        image1_before = chunks[0].detach()
        image2_before = chunks[1].detach()
        repeated_image_embeddings_exact = torch.equal(image1_before, image2_before)
        embedding_difference = (image1_before.float() - image2_before.float()).abs()
        embedding_difference_rms = torch.sqrt(torch.mean(embedding_difference.square()))
        image2_rms = torch.sqrt(torch.mean(image2_before.float().square()))
        embedding_relative_rms = embedding_difference_rms / torch.clamp(image2_rms, min=1e-12)
        embedding_mean_cosine = torch.nn.functional.cosine_similarity(
            image1_before.float(),
            image2_before.float(),
            dim=-1,
        ).mean()

        if state["roll_enabled"]:
            image2_after_3d, verification = roll_visual_token_blocks(
                image2_before.unsqueeze(0),
                shift=1,
            )
            image2_after = image2_after_3d.squeeze(0)
            state["apply_count"] = 1
        else:
            image2_after = image2_before
            token_count = int(image2_before.shape[0])
            verification = {
                "shift": 0,
                "token_axis": 1,
                "block_count": 1,
                "token_count_per_block": token_count,
                "hidden_size": int(image2_before.shape[1]),
                "source_index_for_output": list(range(token_count)),
                "exact_roll_verified": True,
                "max_abs_error": 0.0,
            }
        chunks[1] = image2_after
        final_merged = torch.cat(chunks, dim=0)
        if output_kind == "tensor":
            final_container = final_merged
        else:
            output.pooler_output = final_merged
            final_container = output.pooler_output

        final_chunks = list(final_container.split(token_counts, dim=0))
        image1_exact = torch.equal(final_chunks[0], image1_before)
        image2_exact = torch.equal(final_chunks[1], image2_after)
        if not verification["exact_roll_verified"] or verification["max_abs_error"] != 0.0:
            raise RuntimeError(f"sequence-roll source mapping failed: {verification}")
        if not image1_exact or not image2_exact:
            raise RuntimeError(
                f"final visual container mismatch: image1_exact={image1_exact} image2_exact={image2_exact}"
            )

        state.update(
            {
                "applied": bool(state["roll_enabled"]),
                "visual_hook_count": 1,
                "intervention_family": "visual_sequence_roll" if state["roll_enabled"] else "baseline_capture",
                "stage": "qwen_post_spatial_merger_pre_llm_injection",
                "roll_direction": "right" if state["roll_enabled"] else "none",
                "roll_tokens": 1 if state["roll_enabled"] else 0,
                "pixel_equivalent": None if state["roll_enabled"] else 0,
                "image_count": 2,
                "image_token_counts": token_counts,
                "qwen_grid_thw": [list(map(int, row)) for row in grid_rows],
                "qwen_spatial_merge_size": merge_size,
                "visual_output_kind": output_kind,
                "input_shape": list(merged_output.shape),
                "output_shape": list(final_container.shape),
                "dtype": str(merged_output.dtype),
                "repeated_image_embeddings_exact": bool(repeated_image_embeddings_exact),
                "repeated_image_embeddings_max_abs_diff": float(embedding_difference.max().item()),
                "repeated_image_embeddings_mean_abs_diff": float(embedding_difference.mean().item()),
                "repeated_image_embeddings_relative_rms": float(embedding_relative_rms.item()),
                "repeated_image_embeddings_mean_cosine": float(embedding_mean_cosine.item()),
                "image1_unchanged_exact": bool(image1_exact),
                "image2_final_exact": bool(image2_exact),
                "image1_before_sha256": tensor_sha256(image1_before),
                "image1_after_sha256": tensor_sha256(final_chunks[0]),
                "image2_before_sha256": tensor_sha256(image2_before),
                "image2_after_sha256": tensor_sha256(final_chunks[1]),
                **verification,
                "_image1_before": image1_before,
                "_image1_after": final_chunks[0].detach(),
                "_image2_before": image2_before,
                "_image2_after": final_chunks[1].detach(),
            }
        )
        if output_kind == "tensor":
            return final_merged
        return output

    def _language_pre_hook(self, module, args, kwargs):
        state = self._active
        if state is None:
            return None
        if state["language_injection_hook_count"] != 0:
            raise RuntimeError("Qwen language-model injection hook fired more than once for one flow-probe sample")
        inputs_embeds = kwargs.get("inputs_embeds")
        if not isinstance(inputs_embeds, torch.Tensor) or inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
            raise TypeError(
                "Qwen language model must receive batch-one inputs_embeds for flow-probe validation, "
                f"got {type(inputs_embeds)} shape={getattr(inputs_embeds, 'shape', None)}"
            )
        image1_positions = torch.as_tensor(state["_image1_positions"], device=inputs_embeds.device, dtype=torch.long)
        image2_positions = torch.as_tensor(state["_image2_positions"], device=inputs_embeds.device, dtype=torch.long)
        injected_i1 = torch.index_select(inputs_embeds[0], dim=0, index=image1_positions)
        injected_i2 = torch.index_select(inputs_embeds[0], dim=0, index=image2_positions)
        keep_non_i2 = torch.ones(inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.bool)
        keep_non_i2[image2_positions] = False
        non_i2_embeddings = inputs_embeds[0, keep_non_i2]
        injection_i1_exact = torch.equal(injected_i1, state["_image1_after"])
        injection_i2_exact = torch.equal(injected_i2, state["_image2_after"])
        injection_dtype_exact = (
            injected_i1.dtype == state["_image1_after"].dtype
            and injected_i2.dtype == state["_image2_after"].dtype
        )
        injection_shape_exact = (
            injected_i1.shape == state["_image1_after"].shape
            and injected_i2.shape == state["_image2_after"].shape
        )
        if not injection_i1_exact or not injection_i2_exact or not injection_dtype_exact or not injection_shape_exact:
            raise RuntimeError(
                "post-merger visual tensors were not injected into the expected LLM spans: "
                f"I1={injection_i1_exact} I2={injection_i2_exact} "
                f"dtype={injection_dtype_exact} shape={injection_shape_exact}"
            )
        position_ids = kwargs.get("position_ids")
        attention_mask = kwargs.get("attention_mask")
        state.update(
            {
                "language_injection_hook_count": 1,
                "llm_injection_stage": "qwen_language_model_inputs_embeds",
                "llm_inputs_embeds_shape": list(inputs_embeds.shape),
                "llm_i1_injection_exact": bool(injection_i1_exact),
                "llm_i2_injection_exact": bool(injection_i2_exact),
                "llm_injection_dtype_exact": bool(injection_dtype_exact),
                "llm_injection_shape_exact": bool(injection_shape_exact),
                "llm_inputs_embeds_dtype": str(inputs_embeds.dtype),
                "llm_inputs_embeds_device": str(inputs_embeds.device),
                "llm_i1_sha256": tensor_sha256(injected_i1),
                "llm_i2_sha256": tensor_sha256(injected_i2),
                "llm_non_i2_shape": list(non_i2_embeddings.shape),
                "llm_non_i2_dtype": str(non_i2_embeddings.dtype),
                "llm_non_i2_sha256": tensor_sha256(non_i2_embeddings),
                "position_ids_sha256": tensor_sha256(position_ids) if isinstance(position_ids, torch.Tensor) else None,
                "position_ids_shape": list(position_ids.shape) if isinstance(position_ids, torch.Tensor) else None,
                "position_ids_dtype": str(position_ids.dtype) if isinstance(position_ids, torch.Tensor) else None,
                "attention_mask_sha256": tensor_sha256(attention_mask) if isinstance(attention_mask, torch.Tensor) else None,
                "attention_mask_shape": list(attention_mask.shape) if isinstance(attention_mask, torch.Tensor) else None,
                "attention_mask_dtype": str(attention_mask.dtype) if isinstance(attention_mask, torch.Tensor) else None,
                "_injected_i1": injected_i1.detach(),
                "_injected_i2": injected_i2.detach(),
            }
        )
        return None

    def _finalize_state(self, state: dict[str, Any]) -> None:
        if state["roll_enabled"] and self.dump_count < self.raw_dump_limit:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            raw_path = self.dump_dir / f"{state['case_id']}__{VISUAL_SEQUENCE_ROLL_RIGHT_1}.npz"
            np.savez_compressed(
                raw_path,
                image1_before=state["_image1_before"].float().cpu().numpy(),
                image1_after=state["_image1_after"].float().cpu().numpy(),
                image2_before=state["_image2_before"].float().cpu().numpy(),
                image2_after=state["_image2_after"].float().cpu().numpy(),
                llm_injected_image1=state["_injected_i1"].float().cpu().numpy(),
                llm_injected_image2=state["_injected_i2"].float().cpu().numpy(),
                image1_before_raw_bytes=tensor_raw_bytes(state["_image1_before"]),
                image1_after_raw_bytes=tensor_raw_bytes(state["_image1_after"]),
                image2_before_raw_bytes=tensor_raw_bytes(state["_image2_before"]),
                image2_after_raw_bytes=tensor_raw_bytes(state["_image2_after"]),
                llm_injected_image1_raw_bytes=tensor_raw_bytes(state["_injected_i1"]),
                llm_injected_image2_raw_bytes=tensor_raw_bytes(state["_injected_i2"]),
                source_index_for_output=np.asarray(state["source_index_for_output"], dtype=np.int64),
            )
            state["raw_npz_path"] = str(raw_path.relative_to(self.dump_dir.parent))
            self.dump_count += 1
        for key in [item for item in state if item.startswith("_")]:
            state.pop(key, None)

    def close(self) -> None:
        if self._visual_hook_handle is not None:
            self._visual_hook_handle.remove()
            self._visual_hook_handle = None
        if self._language_hook_handle is not None:
            self._language_hook_handle.remove()
            self._language_hook_handle = None
