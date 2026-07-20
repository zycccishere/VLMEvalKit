from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


NO_VISUAL_TOKEN_SHIFT = "none"
VLLM_NOOP_VISUAL_TOKEN_SHIFT = "noop_vllm"
ROLL_RIGHT_ONE = "roll_right_1"
ROLL_LEFT_ONE = "roll_left_1"
SUPPORTED_VISUAL_TOKEN_SHIFTS = {
    NO_VISUAL_TOKEN_SHIFT,
    VLLM_NOOP_VISUAL_TOKEN_SHIFT,
    ROLL_RIGHT_ONE,
    ROLL_LEFT_ONE,
}
VISUAL_TOKEN_SHIFT_ALIASES = {
    "": NO_VISUAL_TOKEN_SHIFT,
    "baseline": NO_VISUAL_TOKEN_SHIFT,
    "off": NO_VISUAL_TOKEN_SHIFT,
    "noop": VLLM_NOOP_VISUAL_TOKEN_SHIFT,
    "vllm_noop": VLLM_NOOP_VISUAL_TOKEN_SHIFT,
    "right": ROLL_RIGHT_ONE,
    "right_1": ROLL_RIGHT_ONE,
    "roll_right": ROLL_RIGHT_ONE,
    "one_llm_token": ROLL_RIGHT_ONE,
    "left": ROLL_LEFT_ONE,
    "left_1": ROLL_LEFT_ONE,
    "roll_left": ROLL_LEFT_ONE,
}


def _env_truthy(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _safe_token(value: Any) -> str:
    out = []
    for char in str(value):
        out.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    return "".join(out).strip("_") or "sample"


def canonicalize_visual_token_shift(name: str | None, *, strict: bool = True) -> str:
    raw = str(name or NO_VISUAL_TOKEN_SHIFT).strip().lower().replace("-", "_")
    canonical = VISUAL_TOKEN_SHIFT_ALIASES.get(raw, raw)
    if canonical not in SUPPORTED_VISUAL_TOKEN_SHIFTS:
        if strict:
            raise ValueError(f"Unsupported visual token shift: {name}")
        return NO_VISUAL_TOKEN_SHIFT
    return canonical


def visual_token_shift_enabled(name: str | None = None) -> bool:
    if name is None:
        name = os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT", NO_VISUAL_TOKEN_SHIFT)
    return canonicalize_visual_token_shift(name) != NO_VISUAL_TOKEN_SHIFT


def _vllm_recording_arm_path(*, model_family: str) -> Path:
    run_id = os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT_RUN_ID", "").strip()
    dump_dir_raw = os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT_DUMP_DIR", "").strip()
    if not run_id or not dump_dir_raw:
        raise RuntimeError("vLLM visual-token shift recording arm is not configured")
    run_token = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:20]
    return Path(dump_dir_raw) / f"recording_armed.{model_family}.{run_token}.json"


def arm_vllm_visual_token_shift_recording(*, model_family: str) -> Path:
    """Arm evidence recording after vLLM profiling and model initialization finish."""
    path = _vllm_recording_arm_path(model_family=model_family)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "real_request_recording_armed",
        "timestamp": time.time(),
        "run_id": os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT_RUN_ID", ""),
        "model_family": model_family,
        "inference_fingerprint": os.environ.get("REPLAY_INFERENCE_FINGERPRINT", ""),
        "pid": os.getpid(),
    }
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def vllm_visual_token_shift_recording_is_armed(*, model_family: str) -> bool:
    return _vllm_recording_arm_path(model_family=model_family).is_file()


def require_vllm_visual_token_shift_worker_handshake(
    *,
    model_family: str,
    timeout_seconds: float = 30.0,
) -> list[dict[str, Any]]:
    """Require every expected TP worker to instantiate the replay model class."""
    if not visual_token_shift_enabled():
        return []
    run_id = os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT_RUN_ID", "").strip()
    dump_dir_raw = os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT_DUMP_DIR", "").strip()
    target_family = os.environ.get("REPLAY_VLLM_TARGET_FAMILY", "").strip()
    if not run_id or not dump_dir_raw:
        raise RuntimeError("vLLM visual-token shift worker handshake is not configured")
    if target_family != model_family:
        raise RuntimeError(
            f"vLLM visual-token shift target mismatch: {target_family!r} != {model_family!r}"
        )
    tp_size = max(
        1,
        int(
            os.environ.get(
                "VLLM_TP_SIZE",
                os.environ.get("MINICPM45_VLLM_TP_SIZE", "1"),
            )
        ),
    )
    expected_ranks = set(range(tp_size))
    dump_dir = Path(dump_dir_raw)
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        records = []
        for path in dump_dir.glob(f"worker_handshake.{model_family}.pid*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if (
                payload.get("phase") == "worker_model_initialized"
                and payload.get("run_id") == run_id
                and payload.get("model_family") == model_family
                and payload.get("target_family") == target_family
            ):
                records.append(payload)
        observed_ranks = {int(record["rank"]) for record in records}
        if expected_ranks.issubset(observed_ranks):
            return records
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "vLLM replay worker class was not instantiated on every TP rank: "
                f"expected={sorted(expected_ranks)} observed={sorted(observed_ranks)} "
                f"run_id={run_id}"
            )
        time.sleep(0.1)


def shift_for_mode(mode: str) -> int:
    mode = canonicalize_visual_token_shift(mode)
    if mode == ROLL_RIGHT_ONE:
        return 1
    if mode == ROLL_LEFT_ONE:
        return -1
    return 0


def _content_value(item: dict[str, Any]) -> str:
    for key in ("image", "image_url", "video", "value", "path", "text"):
        value = item.get(key)
        if value is not None:
            return str(value)
    return ""


def validate_iqiq_topology(
    *,
    original: list[dict[str, Any]],
    replayed: list[dict[str, Any]],
    replay_mode: str,
    repeat_times: int,
    image_transform_name: str,
    target_image_position: int,
) -> dict[str, Any]:
    """Fail closed unless the payload is exactly one-image IQ -> IQIQ."""
    if str(replay_mode) != "image_text_image_text":
        raise ValueError(f"visual-token shift requires IQIQ replay mode, got {replay_mode!r}")
    if int(repeat_times) != 1:
        raise ValueError(f"visual-token shift requires repeat_times=1, got {repeat_times}")
    if str(image_transform_name) != "baseline":
        raise ValueError(
            "visual-token shift changes embeddings only and cannot be combined with "
            f"REPLAY_IMAGE_TRANSFORM={image_transform_name!r}"
        )
    if int(target_image_position) != 2:
        raise ValueError(f"visual-token shift is defined only for replay I2, got position={target_image_position}")

    def modal(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [item for item in items if isinstance(item, dict) and item.get("type") in {"image", "text", "video"}]

    original_modal = modal(original)
    replayed_modal = modal(replayed)
    original_types = [str(item.get("type")) for item in original_modal]
    replayed_types = [str(item.get("type")) for item in replayed_modal]
    if original_types != ["image", "text"]:
        raise ValueError(f"visual-token shift requires original modal sequence IQ, got {original_types}")
    if replayed_types != ["image", "text", "image", "text"]:
        raise ValueError(f"visual-token shift requires replayed modal sequence IQIQ, got {replayed_types}")

    original_image = _content_value(original_modal[0])
    replay_i1 = _content_value(replayed_modal[0])
    replay_i2 = _content_value(replayed_modal[2])
    original_text = _content_value(original_modal[1])
    replay_q1 = _content_value(replayed_modal[1])
    replay_q2 = _content_value(replayed_modal[3])
    if not original_image or replay_i1 != replay_i2:
        raise ValueError("IQIQ topology check failed: I1 and I2 are not the same replayed image reference")
    if original_image not in {replay_i1, replay_i1.removeprefix("file://")} and replay_i1.removeprefix("file://") != original_image.removeprefix("file://"):
        raise ValueError("IQIQ topology check failed: replayed image does not match the original image")
    if replay_q1 != replay_q2 or original_text != replay_q1:
        raise ValueError("IQIQ topology check failed: Q1 and Q2 are not exact copies of the original question")

    return {
        "validated": True,
        "original_modal_sequence": original_types,
        "replayed_modal_sequence": replayed_types,
        "original_image_count": 1,
        "replayed_image_count": 2,
        "replayed_text_count": 2,
        "i1_i2_same_reference": True,
        "q1_q2_same_text": True,
        "image_transform": "baseline",
        "repeat_times": 1,
    }


def roll_visual_token_blocks(blocks: torch.Tensor, *, shift: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """Roll [block, token, hidden] without changing any shape or value bits."""
    if not isinstance(blocks, torch.Tensor) or blocks.ndim != 3:
        raise ValueError(f"visual token blocks must be [block, token, hidden], got {type(blocks)} {getattr(blocks, 'shape', None)}")
    if blocks.shape[1] < 2:
        raise ValueError(f"visual token series must contain at least two tokens, got {blocks.shape[1]}")
    if shift not in {-1, 0, 1}:
        raise ValueError(f"only no-op or one-token circular shifts are supported, got shift={shift}")

    shifted = torch.roll(blocks, shifts=shift, dims=1)
    token_count = int(blocks.shape[1])
    source_indices = [int((index - shift) % token_count) for index in range(token_count)]
    expected = blocks[:, source_indices, :]
    exact = bool(torch.equal(shifted, expected))
    max_abs_error = float((shifted.float() - expected.float()).abs().max().item())
    return shifted, {
        "shift": int(shift),
        "token_axis": 1,
        "block_count": int(blocks.shape[0]),
        "token_count_per_block": token_count,
        "hidden_size": int(blocks.shape[2]),
        "source_index_for_output": source_indices,
        "exact_roll_verified": exact,
        "max_abs_error": max_abs_error,
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _stable_runtime_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _stable_runtime_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_runtime_value(item) for item in value]
    if isinstance(value, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "type": "numpy.ndarray",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest(),
        }
    if hasattr(value, "mode") and hasattr(value, "size") and hasattr(value, "tobytes"):
        raw = value.tobytes()
        return {
            "type": type(value).__name__,
            "mode": str(value.mode),
            "size": list(value.size),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__, "value": str(value)}


def write_vllm_runtime_contract(
    *,
    model_family: str,
    dataset: str | None,
    requests: list[Any],
    sampling_params: Any,
) -> None:
    if not _env_truthy("REPLAY_VLLM_RUNTIME_CONTRACT", "0"):
        return
    dump_dir_raw = os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT_DUMP_DIR", "").strip()
    run_id = os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT_RUN_ID", "").strip()
    if not dump_dir_raw or not run_id:
        raise RuntimeError("vLLM runtime contract requires dump directory and run id")
    sampling_fields = (
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "stop_token_ids",
        "seed",
    )
    contract_level = os.environ.get("REPLAY_VLLM_RUNTIME_CONTRACT_LEVEL", "full").strip().lower()
    if contract_level not in {"count", "full"}:
        raise RuntimeError(f"unsupported vLLM runtime contract level: {contract_level!r}")
    request_hashes = []
    if contract_level == "full":
        serialized_requests = [_stable_runtime_value(request) for request in requests]
        request_hashes = [
            hashlib.sha256(
                json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            for request in serialized_requests
        ]
    payload = {
        "phase": "vllm_runtime_contract",
        "timestamp": time.time(),
        "run_id": run_id,
        "mode": canonicalize_visual_token_shift(
            os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT", "none")
        ),
        "model_family": model_family,
        "dataset": str(dataset) if dataset is not None else None,
        "request_count": len(requests),
        "request_hashes": request_hashes,
        "request_identity_level": contract_level,
        "effective_infer_batch_size": int(
            os.environ.get("VLMEVAL_EFFECTIVE_INFER_BATCH_SIZE", len(requests))
        ),
        "effective_max_num_seqs": int(os.environ.get("VLLM_MAX_NUM_SEQS", "1")),
        "sampling_contract": {
            field: _stable_runtime_value(getattr(sampling_params, field, None))
            for field in sampling_fields
        },
        "inference_fingerprint": os.environ.get("REPLAY_INFERENCE_FINGERPRINT", ""),
    }
    dump_dir = Path(dump_dir_raw)
    dump_dir.mkdir(parents=True, exist_ok=True)
    path = dump_dir / f"vllm_runtime_contract.pid{os.getpid()}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def roll_vllm_iqiq_image_embeddings(
    multimodal_embeddings: list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    shift: int,
    validate_values: bool = False,
    require_pair_equality: bool = False,
) -> tuple[
    list[torch.Tensor] | tuple[torch.Tensor, ...],
    dict[str, Any],
    list[dict[str, torch.Tensor]],
]:
    """Roll every request-local I2 in flattened ``[I1, I2, ...]`` order.

    vLLM gathers multimodal items in request order before scattering them into
    the language-model input. Chunked prefill is disabled by the wrappers, so
    an IQIQ request contributes exactly two complete image tensors here.
    """
    if not isinstance(multimodal_embeddings, (list, tuple)):
        raise TypeError(
            "visual-token shift requires per-item vLLM embeddings as a list or tuple, "
            f"got {type(multimodal_embeddings)}"
        )
    if not multimodal_embeddings or len(multimodal_embeddings) % 2 != 0:
        raise ValueError(
            "visual-token shift requires a non-empty even sequence [I1, I2] per request, "
            f"got {len(multimodal_embeddings)} item(s)"
        )
    if shift not in {-1, 0, 1}:
        raise ValueError(f"only no-op or one-token circular shifts are supported, got shift={shift}")

    shifted_items = list(multimodal_embeddings)
    pair_records: list[dict[str, Any]] = []
    raw_pairs: list[dict[str, torch.Tensor]] = []
    deferred_pair_equalities: list[tuple[int, torch.Tensor]] = []
    for pair_index in range(len(shifted_items) // 2):
        i1_index = 2 * pair_index
        i2_index = i1_index + 1
        i1 = shifted_items[i1_index]
        i2 = shifted_items[i2_index]
        if not isinstance(i1, torch.Tensor) or not isinstance(i2, torch.Tensor):
            raise TypeError(f"vLLM image pair {pair_index} does not contain tensors")
        if i1.ndim != 2 or i2.ndim != 2:
            raise ValueError(
                f"vLLM image pair {pair_index} must be [token, hidden], "
                f"got {tuple(i1.shape)} and {tuple(i2.shape)}"
            )
        if i1.shape != i2.shape:
            raise ValueError(
                f"vLLM IQIQ pair {pair_index} has different I1/I2 shapes: "
                f"{tuple(i1.shape)} vs {tuple(i2.shape)}"
            )
        if i2.shape[0] < 2:
            raise ValueError(
                f"vLLM IQIQ pair {pair_index} needs at least two visual tokens, got {i2.shape[0]}"
            )
        if i1.dtype != i2.dtype or i1.device != i2.device:
            raise ValueError(
                f"vLLM IQIQ pair {pair_index} dtype/device mismatch: "
                f"{i1.dtype}/{i1.device} vs {i2.dtype}/{i2.device}"
            )

        i1_before = i1.detach().clone() if validate_values else None
        i2_before = i2.detach().clone() if validate_values else None
        same_storage = i1.data_ptr() == i2.data_ptr()
        pair_equal: bool | None = True if same_storage else None
        if validate_values and not same_storage:
            pair_equal = bool(torch.equal(i1, i2))
        elif require_pair_equality and not same_storage:
            deferred_pair_equalities.append((pair_index, torch.eq(i1, i2).all()))
        if pair_equal is False:
            raise ValueError(
                f"vLLM IQIQ pair {pair_index} is not an exact duplicate before token roll"
            )

        shifted_i2 = torch.roll(i2, shifts=shift, dims=0)
        shifted_items[i2_index] = shifted_i2
        source_indices = [
            int((output_index - shift) % i2.shape[0])
            for output_index in range(i2.shape[0])
        ]
        pair_record = {
            "pair_index": pair_index,
            "i1_item_index": i1_index,
            "i2_item_index": i2_index,
            "shape": list(i2.shape),
            "dtype": str(i2.dtype),
            "device": str(i2.device),
            "shift": shift,
            "token_axis": 0,
            "source_index_for_output": source_indices,
            "i1_i2_same_storage_before": same_storage,
            "i1_i2_equal_exact": pair_equal if (validate_values or require_pair_equality) else None,
            "i2_out_of_place": shifted_i2.data_ptr() != i2.data_ptr(),
        }
        if validate_values:
            assert i1_before is not None and i2_before is not None
            expected = i2_before[source_indices]
            pair_record.update(
                {
                    "i1_unchanged_exact": bool(torch.equal(i1, i1_before)),
                    "i2_source_unchanged_exact": bool(torch.equal(i2, i2_before)),
                    "i2_roll_exact": bool(torch.equal(shifted_i2, expected)),
                    "max_abs_error": float(
                        (shifted_i2.float() - expected.float()).abs().max().item()
                    ),
                }
            )
            if not all(
                pair_record[key]
                for key in (
                    "i1_unchanged_exact",
                    "i2_source_unchanged_exact",
                    "i2_roll_exact",
                )
            ):
                raise RuntimeError(f"vLLM visual-token shift invariant failed: {pair_record}")
            raw_pairs.append(
                {
                    "i1_before": i1_before,
                    "i1_after": i1.detach().clone(),
                    "i2_before": i2_before,
                    "i2_after": shifted_i2.detach().clone(),
                    "source_index_for_output": torch.tensor(
                        source_indices,
                        dtype=torch.long,
                        device=i2.device,
                    ),
                }
            )
        pair_records.append(pair_record)

    if deferred_pair_equalities:
        equality_values = torch.stack(
            [value for _, value in deferred_pair_equalities]
        ).detach().cpu().tolist()
        failed_pairs = []
        for (pair_index, _), equal in zip(deferred_pair_equalities, equality_values):
            exact = bool(equal)
            pair_records[pair_index]["i1_i2_equal_exact"] = exact
            if not exact:
                failed_pairs.append(pair_index)
        if failed_pairs:
            raise ValueError(
                "vLLM IQIQ pair(s) are not exact duplicates before token roll: "
                f"{failed_pairs}"
            )

    shifted: list[torch.Tensor] | tuple[torch.Tensor, ...]
    shifted = tuple(shifted_items) if isinstance(multimodal_embeddings, tuple) else shifted_items
    return shifted, {
        "item_count": len(shifted_items),
        "request_pair_count": len(pair_records),
        "container_type": type(multimodal_embeddings).__name__,
        "pair_records": pair_records,
        "value_validation_enabled": validate_values,
        "pair_equality_required": require_pair_equality,
    }, raw_pairs


class _MiniCPMProcessorCaptureProxy:
    def __init__(self, processor: Any, controller: "VisualTokenShiftController") -> None:
        object.__setattr__(self, "_processor", processor)
        object.__setattr__(self, "_controller", controller)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_processor"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_processor", "_controller"}:
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_processor"), name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = object.__getattribute__(self, "_processor")(*args, **kwargs)
        object.__getattribute__(self, "_controller").capture_minicpm_processor_payload(result)
        return result


class VisualTokenShiftController:
    """Owns one-sample shift state and raw dumps for an HF model hook."""

    def __init__(self, *, model_family: str, model_name: str) -> None:
        self.mode = canonicalize_visual_token_shift(os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT", NO_VISUAL_TOKEN_SHIFT))
        self.shift = shift_for_mode(self.mode)
        self.target_image_position = max(1, int(os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT_TARGET_POSITION", "2")))
        self.strict = _env_truthy("REPLAY_VISUAL_TOKEN_SHIFT_STRICT", "1")
        self.raw_dump = _env_truthy("REPLAY_VISUAL_TOKEN_SHIFT_RAW_DUMP", "0")
        self.dump_limit = max(0, int(os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT_DUMP_SAMPLES", "1")))
        dump_dir_raw = (
            os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT_DUMP_DIR")
            or os.environ.get("REPLAY_TRACE_DIR")
            or os.environ.get("REPLAY_DUMP_DIR")
            or ""
        )
        self.dump_dir = Path(dump_dir_raw) if dump_dir_raw else None
        self.model_family = str(model_family)
        self.model_name = str(model_name)
        self._active: dict[str, Any] | None = None
        self._sample_counter = 0
        self._dump_counter = 0
        self._hook_handle = None
        self._original_minicpm_get_vision_embedding = None
        self._jsonl_path: Path | None = None
        if self.enabled and self.dump_dir is not None:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl_path = self.dump_dir / f"visual_token_shift.pid{os.getpid()}.jsonl"

    @property
    def enabled(self) -> bool:
        return self.mode != NO_VISUAL_TOKEN_SHIFT

    @contextlib.contextmanager
    def sample(
        self,
        *,
        dataset: str | None,
        sample_meta: dict[str, Any] | None = None,
        topology: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        if self._active is not None:
            raise RuntimeError("visual token shift sample contexts cannot be nested")
        self._sample_counter += 1
        self._active = {
            "dataset": str(dataset) if dataset is not None else None,
            "sample_meta": dict(sample_meta or {}),
            "sample_counter": self._sample_counter,
            "topology": dict(topology or {}),
            "apply_count": 0,
        }
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            active = self._active
            self._active = None
            if succeeded and self.strict and active is not None and active["apply_count"] != 1:
                raise RuntimeError(
                    f"visual token shift must apply exactly once per successful sample, got {active['apply_count']}"
                )
            if succeeded and self.strict and active is not None and "pending_dump" in active:
                raise RuntimeError("visual token shift was applied but final returned container was not validated")

    def validate_and_bind_iqiq_topology(
        self,
        *,
        original: list[dict[str, Any]],
        replayed: list[dict[str, Any]],
        replay_mode: str,
        repeat_times: int,
        image_transform_name: str,
    ) -> dict[str, Any]:
        topology = validate_iqiq_topology(
            original=original,
            replayed=replayed,
            replay_mode=replay_mode,
            repeat_times=repeat_times,
            image_transform_name=image_transform_name,
            target_image_position=self.target_image_position,
        )
        if self._active is not None:
            self._active["topology"] = topology
        return topology

    def _write_record(
        self,
        record: dict[str, Any],
        *,
        before: torch.Tensor,
        after: torch.Tensor,
        unchanged_before: torch.Tensor | None,
        unchanged_after: torch.Tensor | None,
    ) -> None:
        if self._jsonl_path is None or self._dump_counter >= self.dump_limit:
            return
        self._dump_counter += 1
        stem = "__".join(
            [
                _safe_token(self.model_name),
                _safe_token(record.get("dataset") or "dataset"),
                f"sample{record['sample_counter']}",
                self.mode,
            ]
        )
        if self.dump_dir is None:
            return
        raw_path = self.dump_dir / f"{stem}.npz"
        if self.raw_dump:
            payload = {
                "target_before": before.detach().float().cpu().numpy(),
                "target_after": after.detach().float().cpu().numpy(),
                "source_index_for_output": np.asarray(record["source_index_for_output"], dtype=np.int64),
            }
            if unchanged_before is not None and unchanged_after is not None:
                payload["non_target_before"] = unchanged_before.detach().float().cpu().numpy()
                payload["non_target_after"] = unchanged_after.detach().float().cpu().numpy()
            np.savez_compressed(raw_path, **payload)
            record["raw_npz_path"] = str(raw_path)
        with self._jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _apply_blocks(
        self,
        blocks: torch.Tensor,
        *,
        stage: str,
        target_span: dict[str, Any],
        unchanged_before: torch.Tensor | None,
    ) -> torch.Tensor:
        if self._active is None:
            return blocks
        if self._active["apply_count"] != 0 and self.strict:
            raise RuntimeError("visual token shift hook fired more than once for one sample")
        before = blocks.detach()
        shifted, verification = roll_visual_token_blocks(blocks, shift=self.shift)
        if self.strict and (not verification["exact_roll_verified"] or verification["max_abs_error"] != 0.0):
            raise RuntimeError(f"visual token shift invariant failed: {verification}")
        self._active["apply_count"] += 1
        record = {
            "phase": "visual_token_shift",
            "timestamp": time.time(),
            "model_family": self.model_family,
            "model_name": self.model_name,
            "mode": self.mode,
            "target_image_position": self.target_image_position,
            "stage": stage,
            "dataset": self._active["dataset"],
            "sample_meta": self._active["sample_meta"],
            "sample_counter": self._active["sample_counter"],
            "iqiq_topology": self._active.get("topology", {}),
            "input_shape": list(before.shape),
            "output_shape": list(shifted.shape),
            "input_dtype": str(before.dtype),
            "output_dtype": str(shifted.dtype),
            "input_sha256": _tensor_sha256(before),
            "output_sha256": _tensor_sha256(shifted),
            "token_count_unchanged": before.shape == shifted.shape,
            "dtype_unchanged": before.dtype == shifted.dtype,
            **target_span,
            **verification,
        }
        self._active["pending_dump"] = {
            "record": record,
            "target_before": before,
            "target_after_expected": shifted.detach(),
            "unchanged_before": unchanged_before,
        }
        return shifted

    def _finalize_application(
        self,
        *,
        target_after: torch.Tensor,
        unchanged_after: torch.Tensor | None,
    ) -> None:
        if self._active is None or "pending_dump" not in self._active:
            raise RuntimeError("visual-token shift application has no pending final artifact")
        pending = self._active.pop("pending_dump")
        expected = pending["target_after_expected"]
        unchanged_before = pending["unchanged_before"]
        target_exact = torch.equal(target_after, expected)
        non_target_exact = (
            unchanged_before is None
            and unchanged_after is None
            or unchanged_before is not None
            and unchanged_after is not None
            and torch.equal(unchanged_before, unchanged_after)
        )
        if self.strict and (not target_exact or not non_target_exact):
            raise RuntimeError(
                f"final visual-token container mismatch: target_exact={target_exact} "
                f"non_target_exact={non_target_exact}"
            )
        record = pending["record"]
        record.update(
            {
                "apply_count": int(self._active["apply_count"]),
                "final_target_exact": bool(target_exact),
                "final_non_target_exact": bool(non_target_exact),
                "non_target_before_sha256": _tensor_sha256(unchanged_before)
                if unchanged_before is not None
                else None,
                "non_target_after_sha256": _tensor_sha256(unchanged_after)
                if unchanged_after is not None
                else None,
            }
        )
        self._write_record(
            record,
            before=pending["target_before"],
            after=target_after,
            unchanged_before=unchanged_before,
            unchanged_after=unchanged_after,
        )

    def wrap_minicpm_processor(self, processor: Any) -> Any:
        if isinstance(processor, _MiniCPMProcessorCaptureProxy):
            raise RuntimeError("MiniCPM processor capture is already installed")
        return _MiniCPMProcessorCaptureProxy(processor, self)

    def capture_minicpm_processor_payload(self, data: Any) -> None:
        if self._active is None:
            return
        if "minicpm_processor_payload" in self._active:
            raise RuntimeError("MiniCPM processor payload was captured more than once for one sample")
        image_sizes_batch = data.get("image_sizes")
        image_bounds_batch = data.get("image_bound")
        pixel_values_batch = data.get("pixel_values")
        tgt_sizes_batch = data.get("tgt_sizes")
        if not isinstance(image_sizes_batch, (list, tuple)) or len(image_sizes_batch) != 1:
            raise ValueError(f"MiniCPM processor requires batch=1 image_sizes, got {type(image_sizes_batch)}")
        if not isinstance(image_bounds_batch, (list, tuple)) or len(image_bounds_batch) != 1:
            raise ValueError(f"MiniCPM processor requires batch=1 image_bound, got {type(image_bounds_batch)}")
        if not isinstance(pixel_values_batch, (list, tuple)) or len(pixel_values_batch) != 1:
            raise ValueError(f"MiniCPM processor requires batch=1 pixel_values, got {type(pixel_values_batch)}")
        image_sizes = image_sizes_batch[0]
        image_bounds = image_bounds_batch[0]
        pixel_values = pixel_values_batch[0]
        tgt_sizes = tgt_sizes_batch[0] if isinstance(tgt_sizes_batch, (list, tuple)) else tgt_sizes_batch
        bound_lengths = [int(bound[1] - bound[0]) for bound in image_bounds]
        bound_pairs = [[int(bound[0]), int(bound[1])] for bound in image_bounds]
        payload = {
            "capture_count": 1,
            "image_count": len(image_sizes),
            "image_sizes": [list(map(int, size)) for size in image_sizes],
            "image_bound_count": len(image_bounds),
            "image_bounds": bound_pairs,
            "image_bound_lengths": bound_lengths,
            "pixel_block_count": len(pixel_values),
            "tgt_size_count": len(tgt_sizes),
        }
        if payload["image_count"] != 2 or any(
            payload[key] != 2
            for key in ("image_bound_count", "pixel_block_count", "tgt_size_count")
        ):
            raise ValueError(f"MiniCPM IQIQ processor payload is not one block per replay image: {payload}")
        self._active["minicpm_processor_payload"] = payload

    def install_qwen_hf_hook(self, model: torch.nn.Module) -> None:
        if not self.enabled:
            return
        visual = getattr(model, "visual", None)
        if visual is None and getattr(model, "model", None) is not None:
            visual = getattr(model.model, "visual", None)
        if visual is None:
            raise AttributeError("cannot locate Qwen visual module for visual-token shift")
        if self._hook_handle is not None:
            raise RuntimeError("Qwen visual-token shift hook is already installed")
        if getattr(visual, "_replay_visual_token_shift_installed", False):
            raise RuntimeError("Qwen model instance already has a visual-token shift hook")
        setattr(visual, "_replay_visual_token_shift_installed", True)

        def hook(module, args, kwargs, output):
            if self._active is None:
                return output
            if self.strict and not self._active.get("topology", {}).get("validated"):
                raise RuntimeError("Qwen visual hook fired before IQIQ topology validation")
            if isinstance(output, torch.Tensor):
                merged_output = output
                output_kind = "tensor"
            else:
                merged_output = getattr(output, "pooler_output", None)
                output_kind = "base_model_output_with_pooling"
            if not isinstance(merged_output, torch.Tensor) or merged_output.ndim != 2:
                raise TypeError(
                    "expected Qwen final post-merger pooler_output [tokens, hidden], "
                    f"got output={type(output)} pooler_output={type(merged_output)} "
                    f"shape={getattr(merged_output, 'shape', None)}"
                )
            grid_thw = kwargs.get("grid_thw")
            if grid_thw is None and len(args) >= 2:
                grid_thw = args[1]
            if grid_thw is None:
                raise ValueError("Qwen visual hook did not receive grid_thw")
            grid_rows = grid_thw.detach().cpu().tolist() if hasattr(grid_thw, "detach") else list(grid_thw)
            visual_config = getattr(module, "config", None)
            merge_size = int(
                getattr(module, "spatial_merge_size", 0)
                or getattr(visual_config, "spatial_merge_size", 0)
            )
            if merge_size <= 0:
                raise ValueError(f"invalid Qwen spatial_merge_size={merge_size}")
            sizes = [int(t * h * w // (merge_size * merge_size)) for t, h, w in grid_rows]
            if sum(sizes) != int(merged_output.shape[0]):
                raise ValueError(f"Qwen visual split mismatch: sizes={sizes}, output={list(merged_output.shape)}")
            if len(sizes) != 2 or sizes[0] != sizes[1] or grid_rows[0] != grid_rows[1]:
                raise ValueError(
                    "Qwen visual-token shift requires exactly two identical replay-image grids, "
                    f"got sizes={sizes}, grid_thw={grid_rows}"
                )
            target_index = self.target_image_position - 1
            if target_index >= len(sizes):
                raise ValueError(f"Qwen prompt has {len(sizes)} images; cannot shift image {self.target_image_position}")
            chunks = list(merged_output.split(sizes, dim=0))
            unchanged = chunks[0].detach() if target_index != 0 else (chunks[1].detach() if len(chunks) > 1 else None)
            start = sum(sizes[:target_index])
            target = chunks[target_index]
            shifted = self._apply_blocks(
                target.unsqueeze(0),
                stage="qwen_post_spatial_merger_pre_llm_injection",
                target_span={
                    "image_count": len(sizes),
                    "image_token_counts": sizes,
                    "target_flat_start": start,
                    "target_flat_end_exclusive": start + sizes[target_index],
                    "qwen_grid_thw": [list(map(int, row)) for row in grid_rows],
                    "qwen_spatial_merge_size": merge_size,
                    "qwen_visual_output_kind": output_kind,
                },
                unchanged_before=unchanged.unsqueeze(0) if unchanged is not None else None,
            ).squeeze(0)
            chunks[target_index] = shifted
            final_merged = torch.cat(chunks, dim=0)
            if output_kind == "tensor":
                final_container = final_merged
            else:
                output.pooler_output = final_merged
                final_container = output.pooler_output
            final_chunks = list(final_container.split(sizes, dim=0))
            final_unchanged = final_chunks[0] if target_index != 0 else final_chunks[1]
            self._finalize_application(
                target_after=final_chunks[target_index].unsqueeze(0),
                unchanged_after=final_unchanged.unsqueeze(0),
            )
            if output_kind == "tensor":
                return final_merged
            return output

        self._hook_handle = visual.register_forward_hook(hook, with_kwargs=True)

    def install_minicpm_hf_hook(self, model: torch.nn.Module) -> None:
        if not self.enabled:
            return
        original = getattr(model, "get_vision_embedding", None)
        if not callable(original):
            raise AttributeError("cannot locate MiniCPM get_vision_embedding for visual-token shift")
        if self._original_minicpm_get_vision_embedding is not None:
            raise RuntimeError("MiniCPM visual-token shift hook is already installed")
        if getattr(model, "_replay_visual_token_shift_installed", False):
            raise RuntimeError("MiniCPM model instance already has a visual-token shift hook")
        setattr(model, "_replay_visual_token_shift_installed", True)
        self._original_minicpm_get_vision_embedding = original

        def wrapped(data):
            states = original(data)
            if self._active is None:
                return states
            if self.strict and not self._active.get("topology", {}).get("validated"):
                raise RuntimeError("MiniCPM visual hook fired before IQIQ topology validation")
            if not isinstance(states, list) or len(states) != 1 or not isinstance(states[0], torch.Tensor):
                raise TypeError(f"MiniCPM HF token shift requires batch=1 tensor states, got {type(states)}")
            tensor = states[0]
            if tensor.ndim != 3:
                raise ValueError(f"MiniCPM vision states must be [slice, token, hidden], got {list(tensor.shape)}")
            query_num = int(getattr(getattr(model, "config", None), "query_num", 0))
            if query_num != 64 or int(tensor.shape[1]) != query_num:
                raise ValueError(
                    f"MiniCPM-o-4.5 requires [block, 64, hidden] after resampler, "
                    f"got query_num={query_num}, states={list(tensor.shape)}"
                )
            image_bounds_batch = data.get("image_bound")
            if not isinstance(image_bounds_batch, (list, tuple)) or len(image_bounds_batch) != 1:
                raise ValueError(f"MiniCPM token shift requires batch=1 image_bound, got {type(image_bounds_batch)}")
            image_bounds = image_bounds_batch[0]
            processor_payload = self._active.get("minicpm_processor_payload")
            if not isinstance(processor_payload, dict):
                raise RuntimeError("MiniCPM visual hook fired without captured pre-pop processor payload")
            image_count = int(processor_payload["image_count"])
            block_count = len(image_bounds)
            hook_bound_pairs = [[int(bound[0]), int(bound[1])] for bound in image_bounds]
            if image_count != 2 or block_count != image_count or block_count != int(tensor.shape[0]):
                raise ValueError(
                    "MiniCPM token shift currently supports the verified max_slice_nums=1 payload only: "
                    f"images={image_count}, bounds={block_count}, states={list(tensor.shape)}"
                )
            bound_lengths = [int(bound[1] - bound[0]) for bound in image_bounds]
            if bound_lengths != [query_num, query_num]:
                raise ValueError(f"MiniCPM image placeholder lengths must be [64, 64], got {bound_lengths}")
            if processor_payload.get("capture_count") != 1:
                raise ValueError(f"MiniCPM processor capture count must be one: {processor_payload}")
            if processor_payload.get("image_bounds") != hook_bound_pairs:
                raise ValueError(
                    "MiniCPM image bounds changed between processor capture and vision hook: "
                    f"captured={processor_payload.get('image_bounds')} hook={hook_bound_pairs}"
                )
            if processor_payload.get("image_bound_lengths") != bound_lengths:
                raise ValueError(
                    "MiniCPM image bound lengths changed between processor capture and vision hook: "
                    f"captured={processor_payload.get('image_bound_lengths')} hook={bound_lengths}"
                )
            slice_counts = [1, 1]
            target_index = self.target_image_position - 1
            if target_index >= len(slice_counts):
                raise ValueError(f"MiniCPM prompt has {len(slice_counts)} images; cannot shift image {self.target_image_position}")
            start = sum(slice_counts[:target_index])
            end = start + slice_counts[target_index]
            unchanged = tensor[:start].detach() if start > 0 else tensor[end:].detach()
            shifted_target = self._apply_blocks(
                tensor[start:end],
                stage="minicpm_post_resampler_pre_llm_injection",
                target_span={
                    "image_count": len(slice_counts),
                    "image_slice_counts": slice_counts,
                    "image_placeholder_lengths": bound_lengths,
                    "minicpm_query_num": query_num,
                    "minicpm_processor_payload": processor_payload,
                    "target_slice_start": start,
                    "target_slice_end_exclusive": end,
                    "roll_scope": "each_resampler_block_independently",
                },
                unchanged_before=unchanged if unchanged.numel() else None,
            )
            out = list(states)
            out[0] = torch.cat([tensor[:start], shifted_target, tensor[end:]], dim=0)
            final_tensor = out[0]
            final_unchanged = final_tensor[:start] if start > 0 else final_tensor[end:]
            self._finalize_application(
                target_after=final_tensor[start:end],
                unchanged_after=final_unchanged if final_unchanged.numel() else None,
            )
            return out

        model.get_vision_embedding = wrapped
