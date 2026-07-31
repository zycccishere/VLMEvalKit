from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


SCHEMA_VERSION = "final_model_input.v1"
STAGE = "final_model_input"

_APPEND_LOCK = threading.Lock()
_IDENTITY_ENV_NAMES = {
    "run_uuid": ("REPLAY_RUN_UUID",),
    "matrix_tag": ("REPLAY_MATRIX_TAG", "VLMEVAL_MATRIX_TAG", "MATRIX_TAG", "MATRIX_NAME"),
    "task_tag": ("REPLAY_TASK_TAG", "VLMEVAL_TASK_TAG", "TASK_TAG"),
    "model_key": ("REPLAY_MODEL_KEY", "VLMEVAL_MODEL_KEY", "MODEL_KEY"),
    "dataset": ("REPLAY_DATASET", "VLMEVAL_DATASET", "DATASET"),
    "condition": ("REPLAY_CONDITION", "VLMEVAL_CONDITION", "CONDITION", "REPLAY_MODE"),
    "canonical_index": (
        "REPLAY_CANONICAL_INDEX",
        "VLMEVAL_CANONICAL_INDEX",
        "CANONICAL_INDEX",
    ),
}


def final_input_dump_enabled() -> bool:
    return bool(os.environ.get("REPLAY_RAW_DUMP_PATH", "").strip())


def new_call_id() -> str:
    return uuid.uuid4().hex


def extract_replay_meta(message: Any) -> dict[str, Any]:
    if not isinstance(message, list):
        return {}
    for item in message:
        if isinstance(item, dict) and isinstance(item.get("replay_meta"), dict):
            return dict(item["replay_meta"])
    return {}


def _sha256_bytes(data: bytes, *, prefix: bytes = b"") -> str:
    digest = hashlib.sha256()
    digest.update(prefix)
    digest.update(data)
    return digest.hexdigest()


def _tensor_bytes(value: Any) -> bytes:
    import torch

    tensor = value.detach().cpu().contiguous()
    return tensor.view(torch.uint8).numpy().tobytes()


def _is_tensor(value: Any) -> bool:
    return (
        value is not None
        and type(value).__module__.split(".", 1)[0] == "torch"
        and hasattr(value, "detach")
        and hasattr(value, "shape")
    )


def _is_numpy_array(value: Any) -> bool:
    return (
        value is not None
        and type(value).__module__.split(".", 1)[0] == "numpy"
        and hasattr(value, "shape")
        and hasattr(value, "tobytes")
    )


def _is_pil_image(value: Any) -> bool:
    return (
        value is not None
        and type(value).__module__.startswith("PIL.")
        and hasattr(value, "mode")
        and hasattr(value, "size")
        and hasattr(value, "tobytes")
    )


def _image_size(width: int, height: int) -> dict[str, int]:
    return {"width": int(width), "height": int(height)}


def _summarize_path_image(raw_path: str) -> dict[str, Any]:
    path_text = raw_path[7:] if raw_path.startswith("file://") else raw_path
    path = Path(path_text).expanduser()
    if not path.is_file():
        return {
            "type": "image_reference",
            "reference": raw_path,
            "size": None,
            "sha256": None,
            "sha256_basis": None,
            "unresolved": True,
        }

    data = path.read_bytes()
    summary: dict[str, Any] = {
        "type": "image_file",
        "reference": raw_path,
        "byte_count": len(data),
        "file_sha256": _sha256_bytes(data),
    }
    try:
        from PIL import Image

        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            prefix = f"PIL\0RGB\0{width}x{height}\0".encode("utf-8")
            summary["mode"] = "RGB"
            summary["size"] = _image_size(width, height)
            summary["sha256"] = _sha256_bytes(image.tobytes(), prefix=prefix)
            summary["sha256_basis"] = "rgb_mode_size_and_decoded_pixels"
    except Exception as err:
        summary["size"] = None
        summary["sha256"] = None
        summary["sha256_basis"] = None
        summary["decode_error"] = f"{type(err).__name__}: {err}"
    return summary


def _summarize_data_uri(value: str) -> dict[str, Any]:
    try:
        header, encoded = value.split(",", 1)
        data = base64.b64decode(encoded, validate=True)
    except Exception as err:
        return {
            "type": "data_uri",
            "reference": value[:80],
            "size": None,
            "sha256": None,
            "sha256_basis": None,
            "decode_error": f"{type(err).__name__}: {err}",
        }
    return {
        "type": "data_uri",
        "mime_header": header,
        "byte_count": len(data),
        "size": None,
        "sha256": _sha256_bytes(data),
        "sha256_basis": "decoded_data_uri_bytes",
    }


def summarize_visual_input(value: Any) -> dict[str, Any]:
    if _is_pil_image(value):
        image = value.convert("RGB")
        width, height = image.size
        prefix = f"PIL\0RGB\0{width}x{height}\0".encode("utf-8")
        return {
            "type": "pil_image",
            "mode": "RGB",
            "size": _image_size(width, height),
            "sha256": _sha256_bytes(image.tobytes(), prefix=prefix),
            "sha256_basis": "rgb_mode_size_and_decoded_pixels",
        }

    if _is_tensor(value):
        shape = [int(x) for x in value.shape]
        dtype = str(value.dtype)
        prefix = f"tensor\0{dtype}\0{shape}\0".encode("utf-8")
        summary = {
            "type": "tensor",
            "shape": shape,
            "dtype": dtype,
            "device": str(value.device),
            "size": _image_size(shape[-1], shape[-2]) if len(shape) >= 2 else None,
            "sha256": _sha256_bytes(_tensor_bytes(value), prefix=prefix),
            "sha256_basis": "dtype_shape_and_contiguous_tensor_bytes",
        }
        return summary

    if _is_numpy_array(value):
        array = value
        shape = [int(x) for x in array.shape]
        dtype = str(array.dtype)
        prefix = f"numpy\0{dtype}\0{shape}\0".encode("utf-8")
        return {
            "type": "numpy_array",
            "shape": shape,
            "dtype": dtype,
            "size": _image_size(shape[-1], shape[-2]) if len(shape) >= 2 else None,
            "sha256": _sha256_bytes(array.tobytes(order="C"), prefix=prefix),
            "sha256_basis": "dtype_shape_and_c_order_array_bytes",
        }

    if isinstance(value, str):
        if value.startswith("data:"):
            return _summarize_data_uri(value)
        return _summarize_path_image(value)

    rendered = repr(value)
    return {
        "type": type(value).__name__,
        "repr": rendered,
        "size": None,
        "sha256": _sha256_bytes(rendered.encode("utf-8")),
        "sha256_basis": "python_repr_only_not_visual_bytes",
        "unresolved": True,
    }


def visual_spec(value: Any, *, modality: str = "image", source_ref: str | None = None) -> dict[str, Any]:
    return {"modality": modality, "value": value, "source_ref": source_ref}


def summarize_content_sequence(items: Sequence[Any]) -> list[dict[str, Any]]:
    """Summarize the ordered multimodal content passed to a consumer API."""

    sequence = []
    visual_position = 0
    for position, item in enumerate(items):
        item_type = None
        text_value = None
        if _is_pil_image(item):
            item_type = "image"
        elif isinstance(item, str):
            item_type = "text"
            text_value = item
        elif isinstance(item, Mapping):
            raw_type = str(item.get("type", "")).lower()
            if raw_type in {"image", "image_pil", "input_image"}:
                item_type = "image"
            elif raw_type in {"text", "input_text"}:
                item_type = "text"
                text_value = item.get("text", item.get("value", ""))
        if item_type is None:
            continue
        entry: dict[str, Any] = {"position": position, "type": item_type}
        if item_type == "image":
            entry["visual_position"] = visual_position
            visual_position += 1
        else:
            rendered = str(text_value or "")
            entry["text_sha256"] = _sha256_bytes(rendered.encode("utf-8"))
        sequence.append(entry)
    return sequence


def _safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if _is_tensor(value):
        summary = summarize_visual_input(value)
        summary["type"] = "tensor_summary"
        return summary
    if _is_numpy_array(value):
        summary = summarize_visual_input(value)
        summary["type"] = "numpy_array_summary"
        return summary
    if _is_pil_image(value):
        summary = summarize_visual_input(value)
        summary["type"] = "pil_image_placeholder"
        return summary
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    return {"type": type(value).__name__, "repr": repr(value)}


def _summarize_processor_inputs(inputs: Any) -> Any:
    if inputs is None:
        return None
    if isinstance(inputs, Mapping):
        return {str(key): _safe_json(value) for key, value in inputs.items()}
    return _safe_json(inputs)


def _first_env(names: Sequence[str]) -> tuple[str | None, str | None]:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value, f"env:{name}"
    return None, None


def _task_identity(
    *,
    dataset: str | None,
    model_key: str | None,
    condition: str | None,
    sample_meta: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str | None], list[str]]:
    identity: dict[str, Any] = {}
    sources: dict[str, str | None] = {}
    for field, env_names in _IDENTITY_ENV_NAMES.items():
        identity[field], sources[field] = _first_env(env_names)

    sample_meta = dict(sample_meta or {})
    fallbacks = {
        "dataset": (dataset or sample_meta.get("dataset_name"), "call_arg:dataset" if dataset else "replay_meta:dataset_name"),
        "model_key": (model_key, "call_arg:model_key"),
        "condition": (condition, "call_arg:condition"),
        "canonical_index": (sample_meta.get("sample_index"), "replay_meta:sample_index"),
    }
    for field, (value, source) in fallbacks.items():
        if identity.get(field) is None and value is not None and str(value).strip():
            identity[field] = str(value)
            sources[field] = source

    missing = [field for field in _IDENTITY_ENV_NAMES if identity.get(field) is None]
    return identity, sources, missing


def _append_jsonl(path: str, payload: Mapping[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    with _APPEND_LOCK:
        fd = os.open(path, flags, 0o600)
        try:
            try:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
            except (ImportError, OSError):
                fcntl = None
            offset = 0
            while offset < len(encoded):
                offset += os.write(fd, encoded[offset:])
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def dump_final_model_input(
    *,
    model_family: str,
    backend: str,
    consumer_api: str,
    text_chat_representation: Any,
    visual_inputs: Sequence[Mapping[str, Any]],
    content_sequence: Sequence[Mapping[str, Any]] | None = None,
    processor_inputs: Any = None,
    dataset: str | None = None,
    model_key: str | None = None,
    condition: str | None = None,
    sample_meta: Mapping[str, Any] | None = None,
    call_id: str | None = None,
    parent_call_id: str | None = None,
    batch_position: int | None = None,
    observability: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    dump_path = os.environ.get("REPLAY_RAW_DUMP_PATH", "").strip()
    if not dump_path:
        return None

    try:
        identity, identity_sources, missing_identity = _task_identity(
            dataset=dataset,
            model_key=model_key,
            condition=condition,
            sample_meta=sample_meta,
        )
        summarized_visuals = []
        for position, item in enumerate(visual_inputs):
            summary = summarize_visual_input(item.get("value"))
            source_ref = item.get("source_ref")
            source_summary = summarize_visual_input(source_ref) if source_ref else None
            summary.update(
                {
                    "position": position,
                    "modality": str(item.get("modality", "image")),
                    "source_ref": source_ref,
                    "source_sha256": source_summary.get("sha256") if source_summary else None,
                    "source_size": source_summary.get("size") if source_summary else None,
                }
            )
            summarized_visuals.append(summary)

        correlation_id = call_id or new_call_id()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "stage": STAGE,
            "time_ns": time.time_ns(),
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            "call_correlation_id": correlation_id,
            "parent_call_id": parent_call_id,
            "batch_position": batch_position,
            "model_family": model_family,
            "backend": backend,
            "consumer_api": consumer_api,
            "task_identity": identity,
            "task_identity_sources": identity_sources,
            "missing_task_identity_fields": missing_identity,
            "replay_meta": _safe_json(dict(sample_meta or {})),
            "text_chat_representation": _safe_json(text_chat_representation),
            "content_sequence": _safe_json(list(content_sequence or [])),
            "visual_input_count": len(summarized_visuals),
            "visual_inputs": summarized_visuals,
            "processor_input_summary": _summarize_processor_inputs(processor_inputs),
            "observability": _safe_json(dict(observability or {})),
        }
        _append_jsonl(dump_path, payload)
        return payload
    except Exception as err:
        logging.warning(
            "final model input dump failed for %s/%s: %s: %s",
            model_family,
            backend,
            type(err).__name__,
            err,
        )
        return None


@contextlib.contextmanager
def observe_bound_method(
    owner: Any,
    method_name: str,
    observer: Callable[[tuple[Any, ...], dict[str, Any]], None],
) -> Iterator[None]:
    """Observe an internal call without changing the default non-dump path."""

    original = getattr(owner, method_name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        observer(args, kwargs)
        return original(*args, **kwargs)

    setattr(owner, method_name, wrapped)
    try:
        yield
    finally:
        setattr(owner, method_name, original)
