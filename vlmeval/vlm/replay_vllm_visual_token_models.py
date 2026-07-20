from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .replay_visual_token_shift import (
    canonicalize_visual_token_shift,
    roll_vllm_iqiq_image_embeddings,
    shift_for_mode,
    vllm_visual_token_shift_recording_is_armed,
)


def _truthy(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _runtime_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _true_span_lengths(mask: torch.Tensor) -> torch.Tensor:
    flat = mask.detach().to(dtype=torch.bool).flatten()
    boundary = torch.zeros(flat.shape[0] + 2, dtype=torch.bool, device=flat.device)
    boundary[1:-1] = flat
    edges = torch.nonzero(boundary[1:] != boundary[:-1], as_tuple=False).flatten()
    return edges[1::2] - edges[::2]


def _group_spans_by_item(
    item_token_counts: list[int],
    span_lengths: list[int],
) -> list[list[int]] | None:
    groups: list[list[int]] = []
    span_index = 0
    for item_count in item_token_counts:
        group = []
        group_total = 0
        while span_index < len(span_lengths) and group_total < item_count:
            span_length = int(span_lengths[span_index])
            group.append(span_length)
            group_total += span_length
            span_index += 1
        if group_total != item_count:
            return None
        groups.append(group)
    if span_index != len(span_lengths):
        return None
    return groups


class _ReplayVisualTokenShiftMixin:
    replay_model_family = "unknown"

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        # vLLM reflects this exact keyword-only signature to select its current
        # model-loader ABI. A generic *args/**kwargs signature is treated as a
        # legacy model and is initialized without vllm_config.
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if not _truthy("REPLAY_VISUAL_TOKEN_SHIFT_CHUNKED_PREFILL_DISABLED", "0"):
            raise RuntimeError(
                "vLLM visual-token shift requires the wrapper to disable chunked prefill"
            )
        if not _truthy("REPLAY_VISUAL_TOKEN_SHIFT_PREFIX_CACHING_DISABLED", "0"):
            raise RuntimeError(
                "vLLM visual-token shift requires the wrapper to disable prefix caching"
            )
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        chunked = getattr(scheduler_config, "enable_chunked_prefill", None)
        if chunked is not False:
            raise RuntimeError(
                "vLLM visual-token shift requires enable_chunked_prefill=False"
            )
        disable_chunked_mm_input = getattr(
            scheduler_config,
            "disable_chunked_mm_input",
            None,
        )
        if disable_chunked_mm_input is not True:
            raise RuntimeError(
                "vLLM visual-token shift requires disable_chunked_mm_input=True"
            )
        cache_config = getattr(vllm_config, "cache_config", None)
        prefix_caching = getattr(cache_config, "enable_prefix_caching", None)
        if prefix_caching is not False:
            raise RuntimeError(
                "vLLM visual-token shift requires enable_prefix_caching=False"
            )
        self._replay_shift_call_count = 0
        self._replay_shift_dump_count = 0
        self._replay_shift_instance_id = f"{time.time_ns()}_{id(self)}"
        self._replay_scheduler_max_num_seqs = getattr(
            scheduler_config,
            "max_num_seqs",
            None,
        )
        self._replay_scheduler_max_num_batched_tokens = getattr(
            scheduler_config,
            "max_num_batched_tokens",
            None,
        )
        self._replay_scheduler_enable_chunked_prefill = chunked
        self._replay_scheduler_disable_chunked_mm_input = disable_chunked_mm_input
        self._replay_cache_enable_prefix_caching = prefix_caching
        self._replay_write_worker_handshake()

    def _replay_write_worker_handshake(self) -> None:
        mode = canonicalize_visual_token_shift(os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT", "none"))
        if mode == "none" and not _truthy(
            "REPLAY_VLLM_MATCHED_TOKEN_ROLL_RUNTIME",
            "0",
        ):
            return
        dump_dir_raw = os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT_DUMP_DIR", "").strip()
        if not dump_dir_raw:
            raise RuntimeError("vLLM visual-token shift requires a worker handshake dump directory")
        model_config = getattr(self, "config", None)
        payload = {
            "phase": "worker_model_initialized",
            "timestamp": time.time(),
            "backend": "vllm",
            "run_id": os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT_RUN_ID", ""),
            "mode": mode,
            "target_family": os.environ.get("REPLAY_VLLM_TARGET_FAMILY", ""),
            "model_family": self.replay_model_family,
            "model_name": str(
                getattr(model_config, "_name_or_path", None)
                or getattr(model_config, "name_or_path", None)
                or type(self).__name__
            ),
            "model_class": type(self).__name__,
            "pid": os.getpid(),
            "rank": _runtime_rank(),
            "instance_id": self._replay_shift_instance_id,
            "inference_fingerprint": os.environ.get("REPLAY_INFERENCE_FINGERPRINT", ""),
            "scheduler_max_num_seqs": self._replay_scheduler_max_num_seqs,
            "scheduler_max_num_batched_tokens": self._replay_scheduler_max_num_batched_tokens,
            "scheduler_enable_chunked_prefill": self._replay_scheduler_enable_chunked_prefill,
            "scheduler_disable_chunked_mm_input": self._replay_scheduler_disable_chunked_mm_input,
            "cache_enable_prefix_caching": self._replay_cache_enable_prefix_caching,
            "vllm_engine_mode": (
                "v0"
                if self.replay_model_family == "qwen2_5_vl"
                and not _truthy("VLLM_USE_V1", "1")
                else "v1"
            ),
        }
        dump_dir = Path(dump_dir_raw)
        dump_dir.mkdir(parents=True, exist_ok=True)
        path = dump_dir / (
            f"worker_handshake.{self.replay_model_family}.pid{os.getpid()}"
            f".rank{payload['rank']}.instance{self._replay_shift_instance_id}.json"
        )
        path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    def _replay_shift_embeddings(
        self,
        multimodal_embeddings: Any,
        *,
        stage: str,
        input_ids: torch.Tensor | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> tuple[Any, dict[str, Any] | None]:
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return multimodal_embeddings, None
        mode = canonicalize_visual_token_shift(os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT", "none"))
        if mode == "none":
            return multimodal_embeddings, None
        if os.environ.get("REPLAY_MODE") != "image_text_image_text":
            raise RuntimeError(
                "vLLM visual-token shift requires REPLAY_MODE=image_text_image_text"
            )
        if int(os.environ.get("REPLAY_TIMES", "1")) != 1:
            raise RuntimeError("vLLM visual-token shift requires REPLAY_TIMES=1")
        if os.environ.get("REPLAY_IMAGE_TRANSFORM", "baseline") != "baseline":
            raise RuntimeError("vLLM visual-token shift cannot be combined with an image transform")
        if int(os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT_TARGET_POSITION", "2")) != 2:
            raise RuntimeError("vLLM visual-token shift is defined only for replay image I2")

        if not vllm_visual_token_shift_recording_is_armed(
            model_family=self.replay_model_family
        ):
            shifted, _, _ = roll_vllm_iqiq_image_embeddings(
                multimodal_embeddings,
                shift=shift_for_mode(mode),
                validate_values=False,
                require_pair_equality=False,
            )
            return shifted, None

        self._replay_shift_call_count += 1
        dump_limit = max(0, int(os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT_DUMP_SAMPLES", "1")))
        dump_dir_raw = os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT_DUMP_DIR", "").strip()
        should_dump = bool(dump_dir_raw) and self._replay_shift_dump_count < dump_limit
        full_validation = _truthy("REPLAY_VISUAL_TOKEN_SHIFT_FULL_VALIDATION", "0")
        validate_values = full_validation and (
            should_dump or _truthy("REPLAY_VISUAL_TOKEN_SHIFT_VALIDATE_VALUES", "0")
        )
        strict = _truthy("REPLAY_VISUAL_TOKEN_SHIFT_STRICT", "1")
        shifted, audit, raw_pairs = roll_vllm_iqiq_image_embeddings(
            multimodal_embeddings,
            shift=shift_for_mode(mode),
            validate_values=validate_values,
            require_pair_equality=strict and full_validation,
        )
        model_config = getattr(self, "config", None)
        pending = {
            "phase": "visual_token_shift",
            "timestamp": time.time(),
            "backend": "vllm",
            "run_id": os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT_RUN_ID", ""),
            "stage": stage,
            "model_family": self.replay_model_family,
            "mode": mode,
            "target_image_position": 2,
            "pid": os.getpid(),
            "rank": _runtime_rank(),
            "call_index": self._replay_shift_call_count,
            "real_request": True,
            "recording_armed": True,
            "inference_fingerprint": os.environ.get("REPLAY_INFERENCE_FINGERPRINT", ""),
            "full_tensor_validation": full_validation,
            "validation_level": "full" if full_validation else "lightweight",
            "strict": strict,
            "model_name": str(
                getattr(model_config, "_name_or_path", None)
                or getattr(model_config, "name_or_path", None)
                or type(self).__name__
            ),
            "model_metadata": {
                "image_token_id": getattr(model_config, "image_token_id", None),
                "query_num": getattr(model_config, "query_num", None),
            },
            "audit": audit,
            "dump_dir": dump_dir_raw,
            "should_dump": should_dump,
            "raw_dump": _truthy("REPLAY_VISUAL_TOKEN_SHIFT_RAW_DUMP", "0"),
            "raw_pairs": raw_pairs,
            "input_ids_before": (
                input_ids.detach().clone()
                if full_validation and input_ids is not None
                else None
            ),
            "is_multimodal_before": (
                is_multimodal.detach().clone()
                if full_validation and is_multimodal is not None
                else None
            ),
        }
        return shifted, pending

    def _replay_finalize_shift(
        self,
        pending: dict[str, Any] | None,
        *,
        input_ids: torch.Tensor | None,
        is_multimodal: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        shifted_embeddings: Any = None,
    ) -> None:
        if pending is None:
            return
        should_dump = bool(pending["should_dump"])
        input_before = pending.pop("input_ids_before")
        mask_before = pending.pop("is_multimodal_before")
        raw_pairs = pending.pop("raw_pairs")
        if not pending["full_tensor_validation"]:
            if (
                input_ids is None
                or is_multimodal is None
                or inputs_embeds is None
                or shifted_embeddings is None
            ):
                raise RuntimeError("vLLM visual-token shift cannot audit the lightweight LLM input")
            output_shape_matches = tuple(inputs_embeds.shape[:-1]) == tuple(input_ids.shape)
            if not output_shape_matches:
                raise RuntimeError(
                    "vLLM lightweight output shape mismatch: "
                    f"input_ids={tuple(input_ids.shape)} embeds={tuple(inputs_embeds.shape)}"
                )
            item_token_counts = [int(item.shape[0]) for item in shifted_embeddings]
            pending.update(
                {
                    "input_ids_unchanged_exact": None,
                    "is_multimodal_unchanged_exact": None,
                    "item_token_coverage_exact": None,
                    "item_span_lengths_exact": None,
                    "item_span_lengths_required": None,
                    "item_span_grouping_exact": None,
                    "item_span_groups": None,
                    "final_mm_scatter_exact": None,
                    "output_shape_matches_input_ids": True,
                    "item_token_counts": item_token_counts,
                    "multimodal_span_lengths": None,
                    "final_mm_token_count": None,
                    "expected_mm_token_count": sum(item_token_counts),
                }
            )
            dump_dir = Path(pending["dump_dir"])
            dump_dir.mkdir(parents=True, exist_ok=True)
            pair_records = pending["audit"]["pair_records"]
            call_record = {
                "phase": "visual_token_shift_call",
                "timestamp": pending["timestamp"],
                "backend": "vllm",
                "run_id": pending["run_id"],
                "stage": pending["stage"],
                "model_family": pending["model_family"],
                "model_name": pending["model_name"],
                "mode": pending["mode"],
                "pid": pending["pid"],
                "rank": pending["rank"],
                "call_index": pending["call_index"],
                "real_request": True,
                "recording_armed": True,
                "inference_fingerprint": pending["inference_fingerprint"],
                "full_tensor_validation": False,
                "validation_level": "lightweight",
                "request_pair_count": pending["audit"]["request_pair_count"],
                "all_iqiq_pairs_equal_exact": None,
                "pair_structure_validated": bool(pair_records),
                "output_shape_matches_input_ids": True,
                "expected_mm_token_count": pending["expected_mm_token_count"],
                "qwen_grid_token_count_exact": pending.get(
                    "qwen_grid_token_count_exact"
                ),
            }
            call_path = dump_dir / f"visual_token_shift_calls.vllm.pid{os.getpid()}.jsonl"
            with call_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(call_record, ensure_ascii=False) + "\n")
            if should_dump:
                pending.pop("dump_dir")
                pending.pop("raw_dump")
                pending.pop("should_dump")
                jsonl_path = dump_dir / f"visual_token_shift.vllm.pid{os.getpid()}.jsonl"
                with jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(pending, ensure_ascii=False) + "\n")
                self._replay_shift_dump_count += 1
            return
        if (
            input_before is None
            or mask_before is None
            or input_ids is None
            or is_multimodal is None
            or inputs_embeds is None
            or shifted_embeddings is None
        ):
            raise RuntimeError("vLLM visual-token shift cannot verify the final LLM input")

        device = inputs_embeds.device
        input_exact_check = torch.eq(input_before, input_ids).all().to(device=device)
        mask_exact_check = torch.eq(mask_before, is_multimodal).all().to(device=device)
        item_token_counts = [int(item.shape[0]) for item in shifted_embeddings]
        span_lengths_tensor = _true_span_lengths(is_multimodal)
        item_counts_tensor = torch.tensor(item_token_counts, device=device, dtype=torch.long)
        mask_token_count = is_multimodal.to(device=device, dtype=torch.long).sum()
        expected_mm_token_count = sum(item_token_counts)
        token_coverage_check = torch.eq(
            torch.tensor(expected_mm_token_count, device=device, dtype=torch.long),
            mask_token_count,
        )
        span_layout_required = self.replay_model_family == "qwen2_5_vl"
        if span_layout_required and item_counts_tensor.shape == span_lengths_tensor.shape:
            span_exact_check = torch.eq(
                item_counts_tensor,
                span_lengths_tensor.to(device=device, dtype=torch.long),
            ).all()
        elif span_layout_required:
            span_exact_check = torch.tensor(False, dtype=torch.bool, device=device)
        else:
            span_exact_check = torch.tensor(True, dtype=torch.bool, device=device)
        final_mm_scatter = inputs_embeds[is_multimodal]
        if final_mm_scatter.shape[0] == expected_mm_token_count:
            offset = 0
            scatter_checks = []
            for item in shifted_embeddings:
                next_offset = offset + int(item.shape[0])
                scatter_checks.append(
                    torch.eq(final_mm_scatter[offset:next_offset], item).all()
                )
                offset = next_offset
            scatter_exact_check = torch.stack(scatter_checks).all()
        else:
            scatter_exact_check = torch.tensor(False, dtype=torch.bool, device=device)
        packed_validation = torch.cat(
            [
                torch.stack(
                    [
                        input_exact_check,
                        mask_exact_check,
                        token_coverage_check,
                        span_exact_check.to(device=device),
                        scatter_exact_check.to(device=device),
                    ]
                ).to(dtype=torch.long),
                span_lengths_tensor.to(device=device, dtype=torch.long),
            ]
        ).detach().cpu().tolist()
        validation_values = [bool(value) for value in packed_validation[:5]]
        multimodal_span_lengths = [int(value) for value in packed_validation[5:]]
        item_span_groups = _group_spans_by_item(
            item_token_counts,
            multimodal_span_lengths,
        )
        item_span_grouping_exact = item_span_groups is not None
        (
            pending["input_ids_unchanged_exact"],
            pending["is_multimodal_unchanged_exact"],
            pending["item_token_coverage_exact"],
            pending["item_span_lengths_exact"],
            pending["final_mm_scatter_exact"],
        ) = validation_values
        pending["item_span_lengths_required"] = span_layout_required
        if not span_layout_required:
            pending["item_span_lengths_exact"] = None
        pending["item_span_grouping_exact"] = item_span_grouping_exact
        pending["item_span_groups"] = item_span_groups
        placeholder_slice_counts = (
            [len(group) for group in item_span_groups]
            if self.replay_model_family == "minicpm_o_4_5"
            and item_span_groups is not None
            else None
        )
        pending["minicpm_placeholder_slice_counts"] = placeholder_slice_counts
        if self.replay_model_family == "minicpm_o_4_5":
            query_num = int((pending.get("model_metadata") or {}).get("query_num") or 0)
            expected_query_num = int(
                os.environ.get("REPLAY_MINICPM_EXPECTED_QUERY_NUM", "64")
            )
            pending["minicpm_placeholder_token_contract_exact"] = bool(
                query_num == expected_query_num
                and len(item_token_counts) == len(placeholder_slice_counts or [])
                and all(
                    len(group) == int(slice_count)
                    and all(int(span_length) == query_num for span_length in group)
                    for group, slice_count in zip(
                        item_span_groups or [],
                        placeholder_slice_counts or [],
                    )
                )
                and all(
                    int(item_count) == query_num * int(slice_count)
                    for item_count, slice_count in zip(
                        item_token_counts,
                        placeholder_slice_counts or [],
                    )
                )
            )
        else:
            pending["minicpm_placeholder_token_contract_exact"] = None
        pending["output_shape_matches_input_ids"] = (
            tuple(inputs_embeds.shape[:-1]) == tuple(input_ids.shape)
        )
        pending["item_token_counts"] = item_token_counts
        pending["multimodal_span_lengths"] = multimodal_span_lengths
        pending["final_mm_token_count"] = int(final_mm_scatter.shape[0])
        pending["expected_mm_token_count"] = expected_mm_token_count
        if (
            not all(validation_values)
            or not item_span_grouping_exact
            or (
                self.replay_model_family == "minicpm_o_4_5"
                and pending["minicpm_placeholder_token_contract_exact"] is not True
            )
        ):
            raise RuntimeError(
                "vLLM final LLM-input invariant failed: "
                f"input={validation_values[0]} mask={validation_values[1]} "
                f"coverage={validation_values[2]} spans={validation_values[3]} "
                f"scatter={validation_values[4]} grouping={item_span_grouping_exact} "
                "minicpm_placeholder_tokens="
                f"{pending['minicpm_placeholder_token_contract_exact']}"
            )

        dump_dir = Path(pending["dump_dir"])
        dump_dir.mkdir(parents=True, exist_ok=True)
        pair_records = pending["audit"]["pair_records"]
        call_record = {
            "phase": "visual_token_shift_call",
            "timestamp": pending["timestamp"],
            "backend": "vllm",
            "run_id": pending["run_id"],
            "stage": pending["stage"],
            "model_family": pending["model_family"],
            "model_name": pending["model_name"],
            "mode": pending["mode"],
            "pid": pending["pid"],
            "rank": pending["rank"],
            "call_index": pending["call_index"],
            "real_request": pending["real_request"],
            "recording_armed": pending["recording_armed"],
            "inference_fingerprint": pending["inference_fingerprint"],
            "full_tensor_validation": True,
            "validation_level": "full",
            "request_pair_count": pending["audit"]["request_pair_count"],
            "all_iqiq_pairs_equal_exact": all(
                pair.get("i1_i2_equal_exact") is True for pair in pair_records
            ),
            "pair_structure_validated": bool(pair_records),
            "input_ids_unchanged_exact": pending["input_ids_unchanged_exact"],
            "is_multimodal_unchanged_exact": pending["is_multimodal_unchanged_exact"],
            "final_mm_scatter_exact": pending["final_mm_scatter_exact"],
            "item_token_coverage_exact": pending.get("item_token_coverage_exact"),
            "item_span_lengths_exact": pending.get("item_span_lengths_exact"),
            "item_span_lengths_required": pending.get("item_span_lengths_required"),
            "item_span_grouping_exact": pending.get("item_span_grouping_exact"),
            "item_span_groups": pending.get("item_span_groups"),
            "output_shape_matches_input_ids": pending.get("output_shape_matches_input_ids"),
            "item_token_counts": pending.get("item_token_counts"),
            "multimodal_span_lengths": pending.get("multimodal_span_lengths"),
            "final_mm_token_count": pending.get("final_mm_token_count"),
            "qwen_grid_token_count_exact": pending.get(
                "qwen_grid_token_count_exact"
            ),
            "minicpm_query_num": (pending.get("model_metadata") or {}).get(
                "query_num"
            ),
            "minicpm_placeholder_slice_counts": pending.get(
                "minicpm_placeholder_slice_counts"
            ),
            "minicpm_placeholder_token_contract_exact": pending.get(
                "minicpm_placeholder_token_contract_exact"
            ),
        }
        call_path = dump_dir / f"visual_token_shift_calls.vllm.pid{os.getpid()}.jsonl"
        with call_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(call_record, ensure_ascii=False) + "\n")

        if not should_dump:
            return

        for pair_record, raw_pair in zip(pending["audit"]["pair_records"], raw_pairs):
            pair_record.update(
                {
                    "i1_before_sha256": _tensor_sha256(raw_pair["i1_before"]),
                    "i1_after_sha256": _tensor_sha256(raw_pair["i1_after"]),
                    "i2_before_sha256": _tensor_sha256(raw_pair["i2_before"]),
                    "i2_after_sha256": _tensor_sha256(raw_pair["i2_after"]),
                }
            )
        pending["input_ids_sha256"] = _tensor_sha256(input_before) if input_before is not None else None
        pending["is_multimodal_sha256"] = _tensor_sha256(mask_before) if mask_before is not None else None
        pending["final_mm_scatter_sha256"] = _tensor_sha256(final_mm_scatter)

        pending.pop("dump_dir")
        raw_dump = bool(pending.pop("raw_dump"))
        pending.pop("should_dump")
        dump_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            f"vllm_{self.replay_model_family}.pid{os.getpid()}.rank{pending['rank']}"
            f".instance{self._replay_shift_instance_id}.call{pending['call_index']}"
        )
        if raw_dump:
            arrays: dict[str, np.ndarray] = {}
            for pair_index, raw_pair in enumerate(raw_pairs):
                for key, value in raw_pair.items():
                    raw_value = value.detach().cpu()
                    prefix = f"pair{pair_index}_{key}"
                    if key == "source_index_for_output":
                        arrays[prefix] = raw_value.numpy()
                        continue
                    arrays[prefix] = raw_value.float().numpy()
                    arrays[f"{prefix}_bytes"] = raw_value.contiguous().view(torch.uint8).numpy()
                    arrays[f"{prefix}_shape"] = np.asarray(raw_value.shape, dtype=np.int64)
            if input_before is not None:
                arrays["input_ids"] = input_before.detach().cpu().numpy()
            if mask_before is not None:
                arrays["is_multimodal"] = mask_before.detach().cpu().numpy()
            if final_mm_scatter is not None:
                final_mm_cpu = final_mm_scatter.detach().cpu()
                arrays["final_mm_scatter"] = final_mm_cpu.float().numpy()
                arrays["final_mm_scatter_bytes"] = final_mm_cpu.contiguous().view(torch.uint8).numpy()
                arrays["final_mm_scatter_shape"] = np.asarray(final_mm_cpu.shape, dtype=np.int64)
            raw_path = dump_dir / f"{stem}.npz"
            np.savez_compressed(raw_path, **arrays)
            pending["raw_npz_path"] = str(raw_path)
        jsonl_path = dump_dir / f"visual_token_shift.vllm.pid{os.getpid()}.jsonl"
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(pending, ensure_ascii=False) + "\n")
        self._replay_shift_dump_count += 1

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        shifted, pending = self._replay_shift_embeddings(
            multimodal_embeddings,
            stage="post_gather_pre_llm_embed_input_ids",
            input_ids=input_ids,
            is_multimodal=is_multimodal,
        )
        output = super().embed_input_ids(
            input_ids,
            multimodal_embeddings=shifted,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )
        self._replay_finalize_shift(
            pending,
            input_ids=input_ids,
            is_multimodal=is_multimodal,
            inputs_embeds=output,
            shifted_embeddings=shifted,
        )
        return output

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
    ) -> torch.Tensor:
        image_token_id = getattr(getattr(self, "config", None), "image_token_id", None)
        is_multimodal = input_ids == image_token_id if image_token_id is not None else None
        shifted, pending = self._replay_shift_embeddings(
            multimodal_embeddings,
            stage="post_gather_pre_llm_get_input_embeddings",
            input_ids=input_ids,
            is_multimodal=is_multimodal,
        )
        output = super().get_input_embeddings(input_ids, shifted)
        self._replay_finalize_shift(
            pending,
            input_ids=input_ids,
            is_multimodal=is_multimodal,
            inputs_embeds=output,
            shifted_embeddings=shifted,
        )
        return output


from vllm.multimodal import MULTIMODAL_REGISTRY  # noqa: E402
from vllm.model_executor.models.qwen2_5_vl import (  # noqa: E402
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
)
class ReplayShiftQwen2_5VL(
    _ReplayVisualTokenShiftMixin,
    Qwen2_5_VLForConditionalGeneration,
):
    replay_model_family = "qwen2_5_vl"

    def _process_image_input(self, image_input: Any) -> Any:
        image_embeddings = super()._process_image_input(image_input)
        legacy_v0 = not _truthy("VLLM_USE_V1", "1") and hasattr(
            Qwen2_5_VLForConditionalGeneration,
            "get_input_embeddings_v0",
        )
        if not legacy_v0:
            return image_embeddings
        shifted, pending = self._replay_shift_embeddings(
            image_embeddings,
            stage="legacy_v0_post_projector_pre_llm",
        )
        if pending is not None:
            grid_thw = image_input.get("image_grid_thw")
            merge_size = int(getattr(self.visual, "spatial_merge_size", 0))
            if grid_thw is None or merge_size <= 0:
                raise RuntimeError("Qwen v0 token roll cannot validate image_grid_thw")
            grid_rows = grid_thw.detach().to(dtype=torch.long).cpu()
            expected_counts = (
                grid_rows.prod(dim=-1) // (merge_size * merge_size)
            ).tolist()
            actual_counts = [int(item.shape[0]) for item in shifted]
            grid_token_count_exact = expected_counts == actual_counts
            if not grid_token_count_exact:
                raise RuntimeError(
                    "Qwen v0 visual-token counts differ from image_grid_thw: "
                    f"expected={expected_counts} actual={actual_counts}"
                )
            pending.update(
                {
                    "qwen_image_grid_thw": grid_rows.tolist(),
                    "qwen_spatial_merge_size": merge_size,
                    "qwen_expected_visual_token_counts": expected_counts,
                    "qwen_actual_visual_token_counts": actual_counts,
                    "qwen_grid_token_count_exact": True,
                }
            )
        if getattr(self, "_replay_legacy_v0_active", False):
            self._replay_legacy_v0_pending = (pending, shifted)
        else:
            raise RuntimeError("legacy Qwen v0 image shift escaped get_input_embeddings_v0")
        return shifted

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        image_input: Any = None,
        video_input: Any = None,
    ) -> torch.Tensor:
        image_token_id = getattr(getattr(self, "config", None), "image_token_id", None)
        if image_token_id is None:
            raise RuntimeError("legacy Qwen v0 token roll cannot locate image placeholder token id")
        input_ids_before = input_ids.detach().clone()
        is_multimodal_before = input_ids == image_token_id
        self._replay_legacy_v0_active = True
        self._replay_legacy_v0_pending = None
        try:
            output = super().get_input_embeddings_v0(
                input_ids,
                image_input=image_input,
                video_input=video_input,
            )
        finally:
            self._replay_legacy_v0_active = False
        pending_shift = self._replay_legacy_v0_pending
        self._replay_legacy_v0_pending = None
        if pending_shift is not None:
            pending, shifted = pending_shift
            is_multimodal = input_ids == image_token_id
            if pending is not None and pending["strict"]:
                pending["input_ids_before"] = input_ids_before
                pending["is_multimodal_before"] = is_multimodal_before
            self._replay_finalize_shift(
                pending,
                input_ids=input_ids,
                is_multimodal=is_multimodal,
                inputs_embeds=output,
                shifted_embeddings=shifted,
            )
        return output


try:
    from vllm.model_executor.models.minicpmo import (
        MiniCPMO4_5,
        MiniCPMODummyInputsBuilder,
        MiniCPMOMultiModalProcessor,
        MiniCPMOProcessingInfo,
    )
except ImportError:
    MiniCPMO4_5 = None
else:

    @MULTIMODAL_REGISTRY.register_processor(
        MiniCPMOMultiModalProcessor,
        info=MiniCPMOProcessingInfo,
        dummy_inputs=MiniCPMODummyInputsBuilder,
    )
    class ReplayShiftMiniCPMO45(
        _ReplayVisualTokenShiftMixin,
        MiniCPMO4_5,
    ):
        replay_model_family = "minicpm_o_4_5"
