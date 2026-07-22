#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_SHIFT = {
    "shift_right_half_vit_token": {
        "dx": 7,
        "processed_shift_pixels": 7,
        "base_pixels": 14,
        "semantic_unit": "vit_patch",
        "dx_tokens": 0.25,
    },
    "shift_right_one_vit_token": {
        "dx": 14,
        "processed_shift_pixels": 14,
        "base_pixels": 14,
        "semantic_unit": "vit_patch",
        "dx_tokens": 0.5,
    },
    "shift_right_one_llm_token": {
        "dx": 28,
        "processed_shift_pixels": 28,
        "base_pixels": 28,
        "semantic_unit": "llm_visual_token",
        "dx_tokens": 1.0,
    },
}
VISUAL_SEQUENCE_ROLL_RIGHT_1 = "visual_sequence_roll_right_1"
VISUAL_SEQUENCE_HOOK_NOOP = "visual_sequence_hook_noop"
EXPECTED_RUNTIME_FILES = {
    "torch_package": (
        "/user/wanzihao/miniconda3/envs/vlmevalkit/lib/python3.10/site-packages/torch/__init__.py",
        "9801e75447c7f585545f989f8a21940b60e0c4cc888effb2f08e739664b4b904",
    ),
    "transformers_package": (
        "/user/wanzihao/miniconda3/envs/vlmevalkit/lib/python3.10/site-packages/transformers/__init__.py",
        "a5c7535f1afd63c5c4d43ae7ff2fd4f51d4b3ad1521340a6500946135ba01af1",
    ),
    "qwen_model_class": (
        "/user/wanzihao/miniconda3/envs/vlmevalkit/lib/python3.10/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
        "2d4d5d1fead50d3adf1b2b7a9fb533d7753b3eed8ddbf5cc635bcd1cd7fcd67e",
    ),
    "qwen_attention_class": (
        "/user/wanzihao/miniconda3/envs/vlmevalkit/lib/python3.10/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
        "2d4d5d1fead50d3adf1b2b7a9fb533d7753b3eed8ddbf5cc635bcd1cd7fcd67e",
    ),
    "qwen_processor_class": (
        "/user/wanzihao/miniconda3/envs/vlmevalkit/lib/python3.10/site-packages/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py",
        "8b6b31276e18face06fe525b0057d388cd1e21d12603f045865c4d8b58136eb1",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate raw smoke artifacts from qwen25vl_shift_flow_probe.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-cases", type=int, default=0)
    parser.add_argument("--row-sum-atol", type=float, default=2e-3)
    parser.add_argument("--metric-atol", type=float, default=5e-4)
    parser.add_argument(
        "--expected-interventions",
        nargs="*",
        default=None,
        help="Exact non-baseline intervention set. Defaults to the run summary for backward compatibility.",
    )
    parser.add_argument("--strict-contract", action="store_true")
    parser.add_argument("--strict-logicvista100", action="store_true")
    parser.add_argument("--require-visual-sequence-raw", action="store_true")
    parser.add_argument("--require-processor-pair-raw", action="store_true")
    parser.add_argument(
        "--require-target-box",
        action="store_true",
        help="Require non-empty target-box token mappings and finite target mass metrics for every layer.",
    )
    parser.add_argument(
        "--require-scalar-raw",
        action="store_true",
        help="In scalar mode, require compact per-query scalar NPZ dumps and validate summary metrics from them.",
    )
    return parser


def add_check(checks: list[dict[str, Any]], failures: list[dict[str, Any]], name: str, ok: bool, detail: Any) -> None:
    item = {"name": name, "ok": bool(ok), "detail": detail}
    checks.append(item)
    if not ok:
        failures.append({"name": name, "detail": detail})


def is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def is_nan_number(value: Any) -> bool:
    try:
        return bool(np.isnan(float(value)))
    except Exception:
        return False


def raw_byte_sha256(array: np.ndarray) -> str:
    raw = np.ascontiguousarray(array, dtype=np.uint8)
    return hashlib.sha256(raw.tobytes()).hexdigest()


def decode_raw_float_bytes(array: np.ndarray, dtype_name: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.ascontiguousarray(array, dtype=np.uint8)
    if dtype_name == "torch.bfloat16":
        words = np.frombuffer(raw.tobytes(), dtype="<u2").astype(np.uint32)
        values = (words << 16).view("<f4")
    elif dtype_name == "torch.float16":
        values = np.frombuffer(raw.tobytes(), dtype="<f2").astype(np.float32)
    elif dtype_name == "torch.float32":
        values = np.frombuffer(raw.tobytes(), dtype="<f4")
    else:
        raise ValueError(f"unsupported raw tensor dtype: {dtype_name}")
    return values.reshape(shape)


def bbox_to_token_indices(*, image_size: list[int] | tuple[int, int], grid: dict[str, Any], bbox_xyxy: list[int]) -> list[int]:
    width, height = [float(v) for v in image_size]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    grid_h = int(grid["llm_grid_h"])
    grid_w = int(grid["llm_grid_w"])
    if width <= 0 or height <= 0 or grid_h <= 0 or grid_w <= 0:
        return []
    token_w = width / grid_w
    token_h = height / grid_h
    indices: list[int] = []
    for row in range(grid_h):
        cy = (row + 0.5) * token_h
        if not (y1 <= cy <= y2):
            continue
        for col in range(grid_w):
            cx = (col + 0.5) * token_w
            if x1 <= cx <= x2:
                indices.append(row * grid_w + col)
    return indices


def content_shifted_bbox_token_indices(
    *,
    image_size: list[int] | tuple[int, int],
    grid: dict[str, Any],
    bbox_xyxy: list[int],
    transform_record: dict[str, Any],
) -> list[int]:
    width, height = [float(v) for v in image_size]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    grid_h = int(grid["llm_grid_h"])
    grid_w = int(grid["llm_grid_w"])
    shift = transform_record.get("shift") or {}
    processed_width = float(shift.get("processed_resized_width") or width)
    processed_height = float(shift.get("processed_resized_height") or height)
    dx = float(shift.get("dx", 0.0))
    dy = float(shift.get("dy", 0.0))
    if processed_width <= 0 or processed_height <= 0 or grid_h <= 0 or grid_w <= 0:
        return []
    token_w = processed_width / grid_w
    token_h = processed_height / grid_h
    indices: list[int] = []
    for row in range(grid_h):
        for col in range(grid_w):
            source_x_proc = ((col + 0.5) * token_w - dx) % processed_width
            source_y_proc = ((row + 0.5) * token_h - dy) % processed_height
            source_x = source_x_proc / processed_width * width
            source_y = source_y_proc / processed_height * height
            if x1 <= source_x <= x2 and y1 <= source_y <= y2:
                indices.append(row * grid_w + col)
    return indices


def indices_valid(indices: list[int], size: int) -> bool:
    return len(indices) == len(set(indices)) and all(0 <= int(idx) < size for idx in indices)


def mean_target_mass(matrix_norm: np.ndarray, query_indices: list[int] | None, key_indices: list[int]) -> float:
    if not key_indices:
        return float("nan")
    if query_indices is None:
        return float(matrix_norm[:, key_indices].sum(axis=-1).mean())
    if not query_indices:
        return float("nan")
    return float(matrix_norm[query_indices][:, key_indices].sum(axis=-1).mean())


def metric_matches(observed: Any, expected: float, atol: float) -> bool:
    try:
        observed_float = float(observed)
    except Exception:
        return False
    return bool(np.isfinite(observed_float) and np.isfinite(expected) and abs(observed_float - expected) <= atol)


def scalar_mean_matches(observed: Any, values: np.ndarray, atol: float) -> bool:
    if values.size == 0:
        return False
    return metric_matches(observed, float(np.asarray(values, dtype=np.float64).mean()), atol)


def array_sha256(value: np.ndarray) -> str:
    raw = np.ascontiguousarray(value).view(np.uint8).tobytes()
    return hashlib.sha256(raw).hexdigest()


def validate_processor_pair_raw(
    raw_path: Path,
    contract: dict[str, Any],
    *,
    image1_tokens: int,
    image2_tokens: int,
) -> tuple[bool, dict[str, Any]]:
    detail: dict[str, Any] = {"path": str(raw_path)}
    try:
        with np.load(raw_path) as raw:
            pixel_values = np.asarray(raw["pixel_values"])
            grid_thw = np.asarray(raw["image_grid_thw"])
        detail.update(
            {
                "pixel_values_shape": list(pixel_values.shape),
                "pixel_values_dtype": str(pixel_values.dtype),
                "grid_thw": grid_thw.tolist(),
                "grid_dtype": str(grid_thw.dtype),
            }
        )
        if (
            pixel_values.ndim != 2
            or grid_thw.shape != (2, 3)
            or not np.issubdtype(grid_thw.dtype, np.integer)
            or bool(np.any(grid_thw <= 0))
        ):
            return False, detail
        patch_row_counts = np.prod(grid_thw.astype(np.int64), axis=1)
        raw_same_grid = np.array_equal(grid_thw[0], grid_thw[1])
        merge_size = int(contract.get("spatial_merge_size", 0))
        merge_area = merge_size * merge_size
        if (
            merge_size <= 0
            or bool(np.any(patch_row_counts % merge_area != 0))
            or int(patch_row_counts.sum()) != int(pixel_values.shape[0])
        ):
            return False, detail
        split_at = int(patch_row_counts[0])
        image1_patch_rows = pixel_values[:split_at]
        image2_patch_rows = pixel_values[split_at:]
        image1_sha = array_sha256(image1_patch_rows)
        image2_sha = array_sha256(image2_patch_rows)
        exact = np.array_equal(image1_patch_rows, image2_patch_rows)
        detail.update(
            {
                "patch_row_counts": patch_row_counts.tolist(),
                "image1_sha256": image1_sha,
                "image2_sha256": image2_sha,
                "patch_rows_exact": bool(exact),
                "same_grid": bool(raw_same_grid),
            }
        )
        ok = (
            exact
            and raw_same_grid
            and np.array_equal(grid_thw, np.asarray(contract.get("grid_thw"), dtype=grid_thw.dtype))
            and patch_row_counts.tolist() == contract.get("patch_row_counts")
            and list(pixel_values.shape) == contract.get("pixel_values_shape")
            and [list(image1_patch_rows.shape), list(image2_patch_rows.shape)]
            == contract.get("image_patch_row_shapes")
            and image1_sha == contract.get("image1_sha256")
            and image2_sha == contract.get("image2_sha256")
            and contract.get("patch_rows_exact") is True
            and contract.get("max_abs_diff") == 0.0
            and contract.get("mean_abs_diff") == 0.0
            and int(patch_row_counts[0] // merge_area) == image1_tokens
            and int(patch_row_counts[1] // merge_area) == image2_tokens
        )
        return bool(ok), detail
    except Exception as exc:
        detail["error"] = f"{type(exc).__name__}: {exc}"
        return False, detail


def correspondence_band_mass(matrix_norm: np.ndarray, cheb_dist: np.ndarray, radius: int) -> float:
    mask = cheb_dist <= radius
    return float(matrix_norm[mask].sum() / max(matrix_norm.shape[0], 1))


def expected_normalized_distance(matrix_norm: np.ndarray, euclid_dist: np.ndarray) -> float:
    max_dist = float(np.max(euclid_dist)) if euclid_dist.size else 0.0
    if max_dist <= 0:
        return 0.0
    return float((matrix_norm * (euclid_dist / max_dist)).sum(axis=-1).mean())


def normalized_row_entropy(matrix_norm: np.ndarray) -> float:
    if matrix_norm.size == 0:
        return float("nan")
    logs = np.log(np.clip(matrix_norm, 1e-8, 1.0))
    entropy = -(matrix_norm * logs).sum(axis=-1)
    denominator = np.log(matrix_norm.shape[1]) if matrix_norm.shape[1] > 1 else 1.0
    return float((entropy / denominator).mean())


def wrapped_distance(values: np.ndarray, centers: np.ndarray, period: int) -> np.ndarray:
    direct = np.abs(values[None, :] - centers[:, None])
    return direct if period <= 0 else np.minimum(direct, period - direct)


def spatial_content_distances(
    *,
    query_rows: np.ndarray,
    query_cols: np.ndarray,
    key_rows: np.ndarray,
    key_cols: np.ndarray,
    grid_h: int,
    grid_w: int,
    transform_record: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    shift = transform_record.get("shift") or {}
    stride = float(shift.get("llm_visual_token_stride") or shift.get("qwen_token_stride") or 28.0)
    source_cols = (query_cols.astype(np.float64) - float(shift.get("dx", 0.0)) / stride) % max(grid_w, 1)
    source_rows = (query_rows.astype(np.float64) - float(shift.get("dy", 0.0)) / stride) % max(grid_h, 1)
    column_distance = wrapped_distance(key_cols.astype(np.float64), source_cols, grid_w)
    row_distance = wrapped_distance(key_rows.astype(np.float64), source_rows, grid_h)
    return np.maximum(row_distance, column_distance), np.sqrt(row_distance**2 + column_distance**2)


def validate(
    output_dir: Path,
    *,
    expected_cases: int,
    row_sum_atol: float,
    metric_atol: float,
    require_target_box: bool,
    require_scalar_raw: bool,
    expected_interventions: list[str] | None,
    strict_contract: bool,
    strict_logicvista100: bool,
    require_visual_sequence_raw: bool,
    require_processor_pair_raw: bool,
) -> dict[str, Any]:
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    raw_sequence_count = 0
    raw_processor_pair_count = 0
    full_layer_count = 0
    summary_layer_count = 0
    seen_sequence_raw_paths: set[str] = set()
    seen_attention_raw_paths: set[str] = set()

    if strict_contract:
        add_check(
            checks,
            failures,
            "strict_metric_tolerance",
            0.0 < metric_atol <= 5e-4,
            metric_atol,
        )
        add_check(checks, failures, "strict_expected_cases", expected_cases > 0, expected_cases)
        add_check(
            checks,
            failures,
            "strict_expected_interventions",
            expected_interventions is not None and len(expected_interventions) > 0,
            expected_interventions,
        )
        runtime_identity = summary.get("runtime_identity") or {}
        runtime_files = runtime_identity.get("runtime_files") or {}
        runtime_files_valid = set(runtime_files) == set(EXPECTED_RUNTIME_FILES) and all(
            runtime_files[name].get("path") == expected_path
            and runtime_files[name].get("sha256") == expected_hash
            for name, (expected_path, expected_hash) in EXPECTED_RUNTIME_FILES.items()
        )
        add_check(
            checks,
            failures,
            "strict_runtime_identity",
            runtime_identity.get("torch_version") == "2.7.1+cu126"
            and runtime_identity.get("transformers_version") == "4.53.3"
            and runtime_identity.get("torch_cuda_version") == "12.6"
            and runtime_identity.get("attention_forward_return_arity") == 3
            and runtime_identity.get("python_version") == "3.10.18"
            and runtime_identity.get("python_executable")
            == "/user/wanzihao/miniconda3/envs/vlmevalkit/bin/python"
            and runtime_identity.get("attention_implementation") == "sdpa"
            and bool(runtime_identity.get("cuda_device_name")),
            runtime_identity,
        )
        add_check(
            checks,
            failures,
            "strict_runtime_module_provenance",
            runtime_files_valid,
            runtime_files,
        )
    if strict_logicvista100:
        canonical_interventions = {
            "shift_right_half_vit_token",
            "shift_right_one_vit_token",
            "shift_right_one_llm_token",
            VISUAL_SEQUENCE_ROLL_RIGHT_1,
        }
        add_check(
            checks,
            failures,
            "logicvista100_exact_interventions",
            expected_interventions is not None
            and len(expected_interventions) == len(canonical_interventions)
            and set(expected_interventions) == canonical_interventions,
            expected_interventions,
        )
        add_check(
            checks,
            failures,
            "logicvista100_model",
            str(summary.get("model_path", "")).rstrip("/").endswith("/Qwen2.5-VL-32B-Instruct"),
            summary.get("model_path"),
        )
        add_check(checks, failures, "logicvista100_seed", summary.get("seed") == 1234, summary.get("seed"))
        add_check(
            checks,
            failures,
            "logicvista100_last4",
            summary.get("attn_layers") == "last4" and summary.get("selected_layers") == [60, 61, 62, 63],
            {"attn_layers": summary.get("attn_layers"), "selected_layers": summary.get("selected_layers")},
        )
        add_check(
            checks,
            failures,
            "logicvista100_dump_and_band",
            summary.get("dump_mode") == "summary"
            and summary.get("full_raw_dump_limit") == 1
            and summary.get("band_radius") == 1,
            {
                "dump_mode": summary.get("dump_mode"),
                "full_raw_dump_limit": summary.get("full_raw_dump_limit"),
                "band_radius": summary.get("band_radius"),
            },
        )
        add_check(
            checks,
            failures,
            "logicvista100_pixel_budget",
            summary.get("qwen_min_pixels") == 1003520 and summary.get("qwen_max_pixels") == 12845056,
            {
                "qwen_min_pixels": summary.get("qwen_min_pixels"),
                "qwen_max_pixels": summary.get("qwen_max_pixels"),
            },
        )
        add_check(checks, failures, "strict_nonempty_cases", bool(summary.get("cases")), summary.get("case_count"))
        add_check(
            checks,
            failures,
            "strict_iqi_identity_contract",
            summary.get("mode") == "image_text_image"
            and summary.get("policy") == "identity"
            and summary.get("text_scope") == "historical_all_non_image_non_special",
            {
                "mode": summary.get("mode"),
                "policy": summary.get("policy"),
                "text_scope": summary.get("text_scope"),
            },
        )

    if expected_cases:
        add_check(checks, failures, "case_count", summary.get("case_count") == expected_cases, summary.get("case_count"))

    expected_non_baseline = list(
        expected_interventions if expected_interventions is not None else summary.get("transforms", [])
    )
    expected_transform_set = {"baseline", *expected_non_baseline}

    selected_layer_ids = [int(value) for value in summary.get("selected_layers", [])]
    for case_index, case in enumerate(summary["cases"]):
        transforms = case["transforms"]
        add_check(
            checks,
            failures,
            f"{case['case_id']} transform_set",
            set(transforms) == expected_transform_set,
            sorted(transforms),
        )
        baseline_tokens = None
        if "baseline" in transforms:
            baseline_tokens = (
                int(transforms["baseline"]["image1_grid"]["token_count"]),
                int(transforms["baseline"]["image2_grid"]["token_count"]),
            )
        for transform, payload in transforms.items():
            image1_tokens = int(payload["image1_grid"]["token_count"])
            image2_tokens = int(payload["image2_grid"]["token_count"])
            add_check(
                checks,
                failures,
                f"{case['case_id']} {transform} repeated_image_token_count_match",
                image1_tokens == image2_tokens,
                {"image1_tokens": image1_tokens, "image2_tokens": image2_tokens},
            )
            if baseline_tokens is not None:
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} {transform} token_count_matches_baseline",
                    (image1_tokens, image2_tokens) == baseline_tokens,
                    {"baseline": baseline_tokens, "current": (image1_tokens, image2_tokens)},
                )
            record = payload["transform_record"]
            intervention = payload.get("intervention") or {}
            sequence_record = intervention.get("visual_sequence_roll") or {}
            is_sequence_roll = transform == VISUAL_SEQUENCE_ROLL_RIGHT_1
            is_hook_noop = transform == VISUAL_SEQUENCE_HOOK_NOOP
            attention_runtime = payload.get("attention_runtime_contract") or {}
            image_pair_contract = payload.get("processed_image_pair_contract") or {}
            payload_layers = payload.get("layers") or []
            payload_layer_ids = [int(layer.get("layer", -1)) for layer in payload_layers]
            add_check(
                checks,
                failures,
                f"{case['case_id']} {transform} exact_layer_set",
                len(payload_layer_ids) == len(selected_layer_ids)
                and sorted(payload_layer_ids) == sorted(selected_layer_ids),
                {"observed": payload_layer_ids, "expected": selected_layer_ids},
            )
            if transform in {"baseline", VISUAL_SEQUENCE_ROLL_RIGHT_1, VISUAL_SEQUENCE_HOOK_NOOP}:
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} {transform} processed_repeated_image_contract",
                    image_pair_contract.get("validated") is True
                    and image_pair_contract.get("stage") == "qwen_processor_output_pre_vit"
                    and image_pair_contract.get("image_count") == 2
                    and image_pair_contract.get("same_grid") is True
                    and image_pair_contract.get("same_shape") is True
                    and image_pair_contract.get("patch_rows_exact") is True
                    and int(image_pair_contract.get("spatial_merge_size", 0)) > 0
                    and len(image_pair_contract.get("patch_row_counts") or []) == 2
                    and image_pair_contract.get("max_abs_diff") == 0.0
                    and image_pair_contract.get("mean_abs_diff") == 0.0
                    and image_pair_contract.get("image1_sha256") == image_pair_contract.get("image2_sha256"),
                    image_pair_contract,
                )
            if strict_contract:
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} {transform} attention_runtime_contract",
                    bool(attention_runtime.get("effective_i2_causal_block_sha256"))
                    and int(attention_runtime.get("effective_i2_causal_future_count", 0)) > 0
                    and bool(attention_runtime.get("mrope_cos_sha256"))
                    and bool(attention_runtime.get("mrope_sin_sha256")),
                    attention_runtime,
                )
            if transform == "baseline":
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} baseline_not_applied",
                    record.get("applied") is False,
                    record,
                )
            elif transform in EXPECTED_SHIFT:
                shift = record.get("shift") or {}
                expected = EXPECTED_SHIFT[transform]
                detail = {
                    key: shift.get(key)
                    for key in [
                        "dx",
                        "dy",
                        "processed_shift_pixels",
                        "base_pixels",
                        "semantic_unit",
                        "processed_space",
                        "border_wrap_verified",
                    ]
                }
                ok = (
                    shift.get("dx") == expected["dx"]
                    and shift.get("dy") == 0
                    and shift.get("processed_shift_pixels") == expected["processed_shift_pixels"]
                    and shift.get("base_pixels") == expected["base_pixels"]
                    and shift.get("semantic_unit") == expected["semantic_unit"]
                    and shift.get("processed_space") is True
                    and shift.get("border_wrap_verified") is True
                )
                add_check(checks, failures, f"{case['case_id']} {transform} shift_record", ok, detail)
                meta = payload.get("content_shift_meta") or {}
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} {transform} dx_tokens",
                    abs(float(meta.get("dx_tokens", -999.0)) - expected["dx_tokens"]) < 1e-8,
                    meta,
                )
            elif is_sequence_roll:
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} {transform} baseline_pixels",
                    record.get("applied") is False,
                    record,
                )
                sequence_detail = {
                    "family": intervention.get("family"),
                    "pixel_equivalent": intervention.get("pixel_equivalent"),
                    "stage": sequence_record.get("stage"),
                    "apply_count": sequence_record.get("apply_count"),
                    "roll_tokens": sequence_record.get("roll_tokens"),
                    "source_index_for_output": sequence_record.get("source_index_for_output"),
                    "raw_npz_path": sequence_record.get("raw_npz_path"),
                }
                source_indices = [int(v) for v in sequence_record.get("source_index_for_output", [])]
                expected_source = [image2_tokens - 1, *range(image2_tokens - 1)]
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} {transform} sequence_contract",
                    intervention.get("family") == "visual_sequence_roll"
                    and intervention.get("pixel_equivalent") is None
                    and sequence_record.get("applied") is True
                    and sequence_record.get("capture_enabled") is True
                    and sequence_record.get("apply_count") == 1
                    and sequence_record.get("stage") == "qwen_post_spatial_merger_pre_llm_injection"
                    and sequence_record.get("roll_tokens") == 1
                    and sequence_record.get("token_axis") == 1
                    and sequence_record.get("exact_roll_verified") is True
                    and sequence_record.get("image1_unchanged_exact") is True
                    and sequence_record.get("image2_final_exact") is True
                    and isinstance(sequence_record.get("repeated_image_embeddings_exact"), bool)
                    and is_finite_number(sequence_record.get("repeated_image_embeddings_max_abs_diff"))
                    and is_finite_number(sequence_record.get("repeated_image_embeddings_mean_abs_diff"))
                    and is_finite_number(sequence_record.get("repeated_image_embeddings_relative_rms"))
                    and is_finite_number(sequence_record.get("repeated_image_embeddings_mean_cosine"))
                    and sequence_record.get("visual_hook_count") == 1
                    and sequence_record.get("language_injection_hook_count") == 1
                    and sequence_record.get("llm_i1_injection_exact") is True
                    and sequence_record.get("llm_i2_injection_exact") is True
                    and sequence_record.get("llm_injection_dtype_exact") is True
                    and sequence_record.get("llm_injection_shape_exact") is True
                    and sequence_record.get("position_ids_sha256")
                    and sequence_record.get("attention_mask_sha256")
                    and sequence_record.get("image1_before_sha256") == sequence_record.get("image1_after_sha256")
                    and intervention.get("matched_baseline_exact") is True
                    and source_indices == expected_source,
                    sequence_detail,
                )
                content_meta = payload.get("content_shift_meta") or {}
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} {transform} spatial_metrics_not_applicable",
                    content_meta.get("mapping_kind") == "visual_sequence_index_roll"
                    and content_meta.get("pixel_equivalent") is None
                    and content_meta.get("spatial_content_metrics_valid") is False,
                    content_meta,
                )
                raw_npz_rel = str(sequence_record.get("raw_npz_path") or "")
                if raw_npz_rel:
                    raw_sequence_count += 1
                    raw_path = output_dir / raw_npz_rel
                    raw = np.load(raw_path)
                    i1_before = np.asarray(raw["image1_before"])
                    i1_after = np.asarray(raw["image1_after"])
                    i2_before = np.asarray(raw["image2_before"])
                    i2_after = np.asarray(raw["image2_after"])
                    injected_i1 = np.asarray(raw["llm_injected_image1"])
                    injected_i2 = np.asarray(raw["llm_injected_image2"])
                    raw_source = np.asarray(raw["source_index_for_output"], dtype=np.int64)
                    raw_tensor_bindings = {
                        "image1_before_raw_bytes": ("image1_before_sha256", i1_before, sequence_record.get("dtype")),
                        "image1_after_raw_bytes": ("image1_after_sha256", i1_after, sequence_record.get("dtype")),
                        "image2_before_raw_bytes": ("image2_before_sha256", i2_before, sequence_record.get("dtype")),
                        "image2_after_raw_bytes": ("image2_after_sha256", i2_after, sequence_record.get("dtype")),
                        "llm_injected_image1_raw_bytes": (
                            "llm_i1_sha256",
                            injected_i1,
                            sequence_record.get("llm_inputs_embeds_dtype"),
                        ),
                        "llm_injected_image2_raw_bytes": (
                            "llm_i2_sha256",
                            injected_i2,
                            sequence_record.get("llm_inputs_embeds_dtype"),
                        ),
                    }
                    raw_hashes: dict[str, str] = {}
                    raw_hash_binding_ok = raw_npz_rel not in seen_sequence_raw_paths
                    seen_sequence_raw_paths.add(raw_npz_rel)
                    for raw_field, (summary_field, float_array, dtype_name) in raw_tensor_bindings.items():
                        if raw_field not in raw:
                            raw_hash_binding_ok = False
                            continue
                        raw_bytes = np.asarray(raw[raw_field])
                        if raw_bytes.dtype != np.uint8 or raw_bytes.ndim != 1:
                            raw_hash_binding_ok = False
                            continue
                        observed_hash = raw_byte_sha256(raw_bytes)
                        raw_hashes[raw_field] = observed_hash
                        try:
                            reconstructed = decode_raw_float_bytes(
                                raw_bytes,
                                str(dtype_name),
                                tuple(float_array.shape),
                            )
                        except (ValueError, TypeError):
                            raw_hash_binding_ok = False
                            continue
                        raw_hash_binding_ok = (
                            raw_hash_binding_ok
                            and observed_hash == sequence_record.get(summary_field)
                            and np.array_equal(reconstructed, float_array.astype(np.float32, copy=False))
                        )
                    embedding_exact = np.array_equal(i1_before, i2_before)
                    embedding_difference = np.abs(
                        i1_before.astype(np.float64) - i2_before.astype(np.float64)
                    )
                    embedding_max_abs_diff = float(embedding_difference.max())
                    embedding_mean_abs_diff = float(embedding_difference.mean())
                    embedding_difference_rms = float(np.sqrt(np.mean(np.square(embedding_difference))))
                    image2_rms = float(np.sqrt(np.mean(np.square(i2_before.astype(np.float64)))))
                    embedding_relative_rms = embedding_difference_rms / max(image2_rms, 1e-12)
                    i1_rows = i1_before.astype(np.float64)
                    i2_rows = i2_before.astype(np.float64)
                    cosine_denominator = np.maximum(np.linalg.norm(i1_rows, axis=-1), 1e-8) * np.maximum(
                        np.linalg.norm(i2_rows, axis=-1),
                        1e-8,
                    )
                    embedding_mean_cosine = float(
                        np.mean(np.sum(i1_rows * i2_rows, axis=-1) / cosine_denominator)
                    )
                    embedding_metrics_match = (
                        sequence_record.get("repeated_image_embeddings_exact") == bool(embedding_exact)
                        and metric_matches(
                            sequence_record.get("repeated_image_embeddings_max_abs_diff"),
                            embedding_max_abs_diff,
                            1e-6,
                        )
                        and metric_matches(
                            sequence_record.get("repeated_image_embeddings_mean_abs_diff"),
                            embedding_mean_abs_diff,
                            1e-6,
                        )
                        and metric_matches(
                            sequence_record.get("repeated_image_embeddings_relative_rms"),
                            embedding_relative_rms,
                            1e-6,
                        )
                        and metric_matches(
                            sequence_record.get("repeated_image_embeddings_mean_cosine"),
                            embedding_mean_cosine,
                            1e-6,
                        )
                        and embedding_max_abs_diff >= embedding_mean_abs_diff >= 0.0
                        and embedding_relative_rms >= 0.0
                        and -1.0 <= embedding_mean_cosine <= 1.0
                        and (
                            not embedding_exact
                            or (
                                embedding_max_abs_diff == 0.0
                                and embedding_mean_abs_diff == 0.0
                                and embedding_relative_rms == 0.0
                            )
                        )
                    )
                    raw_detail = {
                        "path": str(raw_path),
                        "i1_shape": list(i1_before.shape),
                        "i2_shape": list(i2_before.shape),
                        "source_head": raw_source[:8].tolist(),
                        "source_tail": raw_source[-8:].tolist(),
                        "repeated_image_embeddings_exact": bool(embedding_exact),
                        "repeated_image_embeddings_max_abs_diff": embedding_max_abs_diff,
                        "repeated_image_embeddings_mean_abs_diff": embedding_mean_abs_diff,
                        "repeated_image_embeddings_relative_rms": embedding_relative_rms,
                        "repeated_image_embeddings_mean_cosine": embedding_mean_cosine,
                        "embedding_metrics_match": bool(embedding_metrics_match),
                        "raw_hash_binding_ok": bool(raw_hash_binding_ok),
                        "raw_hashes": raw_hashes,
                    }
                    add_check(
                        checks,
                        failures,
                        f"{case['case_id']} {transform} raw_embedding_dump",
                        i1_before.shape == i1_after.shape
                        and i2_before.shape == i2_after.shape
                        and i1_before.shape[0] == image1_tokens
                        and i2_before.shape[0] == image2_tokens
                        and np.array_equal(raw_source, np.asarray(expected_source, dtype=np.int64))
                        and np.array_equal(i1_before, i1_after)
                        and np.array_equal(i2_after, i2_before[raw_source])
                        and np.array_equal(injected_i1, i1_after)
                        and np.array_equal(injected_i2, i2_after)
                        and embedding_metrics_match,
                        raw_detail,
                    )
                    add_check(
                        checks,
                        failures,
                        f"{case['case_id']} {transform} raw_sha256_binding",
                        raw_hash_binding_ok,
                        raw_detail,
                    )
                processor_raw_rel = str(image_pair_contract.get("raw_npz_path") or "")
                if processor_raw_rel:
                    raw_processor_pair_count += 1
                    processor_raw_ok, processor_raw_detail = validate_processor_pair_raw(
                        output_dir / processor_raw_rel,
                        image_pair_contract,
                        image1_tokens=image1_tokens,
                        image2_tokens=image2_tokens,
                    )
                    add_check(
                        checks,
                        failures,
                        f"{case['case_id']} {transform} raw_processor_pair_dump",
                        processor_raw_ok,
                        processor_raw_detail,
                    )
                baseline_payload = transforms.get(VISUAL_SEQUENCE_HOOK_NOOP) or transforms.get("baseline")
                baseline_sequence = (
                    (baseline_payload.get("intervention") or {}).get("visual_sequence_roll") or {}
                    if baseline_payload
                    else {}
                )
                runtime_fields = [
                    "image1_before_sha256",
                    "image2_before_sha256",
                    "qwen_grid_thw",
                    "image_token_counts",
                    "position_ids_sha256",
                    "position_ids_shape",
                    "position_ids_dtype",
                    "attention_mask_sha256",
                    "attention_mask_shape",
                    "attention_mask_dtype",
                    "llm_non_i2_sha256",
                    "llm_non_i2_shape",
                    "llm_non_i2_dtype",
                ]
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} {transform} independent_baseline_match",
                    bool(baseline_payload)
                    and payload.get("prompt_text") == baseline_payload.get("prompt_text")
                    and payload.get("model_input_fingerprint") == baseline_payload.get("model_input_fingerprint")
                    and payload.get("attention_runtime_contract") == baseline_payload.get("attention_runtime_contract")
                    and all(sequence_record.get(field) == baseline_sequence.get(field) for field in runtime_fields),
                    {"runtime_fields": runtime_fields, "baseline_transform": VISUAL_SEQUENCE_HOOK_NOOP if VISUAL_SEQUENCE_HOOK_NOOP in transforms else "baseline"},
                )
            elif is_hook_noop:
                baseline_payload = transforms.get("baseline") or {}
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} {transform} matched_noop_contract",
                    record.get("applied") is False
                    and intervention.get("family") == "visual_sequence_control"
                    and intervention.get("matched_baseline_exact") is True
                    and sequence_record.get("applied") is False
                    and sequence_record.get("apply_count") == 0
                    and sequence_record.get("visual_hook_count") == 1
                    and sequence_record.get("language_injection_hook_count") == 1
                    and sequence_record.get("llm_i1_injection_exact") is True
                    and sequence_record.get("llm_i2_injection_exact") is True
                    and payload.get("model_input_fingerprint") == baseline_payload.get("model_input_fingerprint")
                    and payload.get("attention_runtime_contract") == baseline_payload.get("attention_runtime_contract"),
                    sequence_record,
                )
            else:
                add_check(
                    checks,
                    failures,
                    f"{case['case_id']} {transform} recognized_intervention",
                    False,
                    sorted([*EXPECTED_SHIFT, VISUAL_SEQUENCE_ROLL_RIGHT_1, VISUAL_SEQUENCE_HOOK_NOOP]),
                )
            for layer in payload_layers:
                layer_name = f"{case['case_id']} {transform} L{layer['layer']}"
                dump_mode = str(layer.get("dump_mode", "full"))
                if strict_logicvista100:
                    expected_dump_mode = "full" if case_index == 0 else "summary"
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} strict_dump_layout",
                        dump_mode == expected_dump_mode,
                        {"observed": dump_mode, "expected": expected_dump_mode, "case_index": case_index},
                    )
                if dump_mode == "summary":
                    summary_layer_count += 1
                    query_count = int(layer.get("query_count", 0) or 0)
                    image1_key_count = int(layer.get("image1_key_count", 0) or 0)
                    image2_key_count = int(layer.get("image2_key_count", 0) or 0)
                    text_key_count = int(layer.get("text_key_count", 0) or 0)
                    core_metric_names = [
                        "position_band_mass",
                        "exact_position_mass",
                        "local_correspondence_band_mass",
                        "expected_position_distance",
                        "expected_distance_from_diagonal",
                        "row_entropy",
                        "mean_image1_mass_raw",
                        "mean_text_mass_raw",
                        "mean_image2_mass_raw",
                        "mass_total_mean",
                        "mass_total_max",
                        "future_i2_mass_max",
                        "i2_total_self_mass_raw",
                        "i2_past_self_mass_raw",
                        "i2_diag_self_mass_raw",
                        "i2_local_self_mass_raw",
                        "i2_local_self_ratio",
                    ]
                    summary_detail = {
                        "query_count": query_count,
                        "image1_key_count": image1_key_count,
                        "text_key_count": text_key_count,
                        "image2_key_count": image2_key_count,
                        "npz_path": layer.get("npz_path"),
                        "core_metrics": {name: layer.get(name) for name in core_metric_names},
                    }
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} summary_counts",
                        query_count == image2_tokens
                        and image1_key_count == image1_tokens
                        and image2_key_count == image2_tokens
                        and text_key_count >= 0,
                        summary_detail,
                    )
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} summary_finite",
                        all(is_finite_number(layer.get(name)) for name in core_metric_names),
                        summary_detail,
                    )
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} summary_causal_and_mass",
                        float(layer.get("future_i2_mass_max", 9.0)) < row_sum_atol
                        and float(layer.get("mass_total_max", 9.0)) <= 1.0 + row_sum_atol,
                        summary_detail,
                    )
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} summary_has_no_matrix_path",
                        not layer.get("npz_path"),
                        summary_detail,
                    )
                    if is_sequence_roll:
                        source_metric_names = [
                            "source_index_band_mass",
                            "source_index_exact_mass",
                            "expected_source_index_distance",
                            "source_minus_position_exact_mass",
                        ]
                        source_minus_expected = float(layer.get("source_index_exact_mass", float("nan"))) - float(
                            layer.get("exact_position_mass", float("nan"))
                        )
                        add_check(
                            checks,
                            failures,
                            f"{layer_name} summary_sequence_metric_semantics",
                            all(is_finite_number(layer.get(name)) for name in source_metric_names)
                            and is_nan_number(layer.get("content_band_mass"))
                            and is_nan_number(layer.get("expected_content_distance"))
                            and metric_matches(
                                layer.get("source_minus_position_exact_mass"),
                                source_minus_expected,
                                1e-10,
                            ),
                            {
                                "source_metrics": {name: layer.get(name) for name in source_metric_names},
                                "content_band_mass": layer.get("content_band_mass"),
                                "expected_content_distance": layer.get("expected_content_distance"),
                                "source_minus_expected": source_minus_expected,
                            },
                        )
                    else:
                        source_metric_names = [
                            "source_index_band_mass",
                            "source_index_exact_mass",
                            "expected_source_index_distance",
                            "source_minus_position_exact_mass",
                        ]
                        add_check(
                            checks,
                            failures,
                            f"{layer_name} summary_spatial_metric_semantics",
                            is_finite_number(layer.get("content_band_mass"))
                            and is_finite_number(layer.get("expected_content_distance"))
                            and all(is_nan_number(layer.get(name)) for name in source_metric_names),
                            {
                                "content_band_mass": layer.get("content_band_mass"),
                                "expected_content_distance": layer.get("expected_content_distance"),
                                "source_metrics": {name: layer.get(name) for name in source_metric_names},
                            },
                        )
                    target_metric_names = [
                        "target_mass_norm_all_queries",
                        "target_mass_norm_target_queries",
                        "target_mass_norm_content_shifted_target_queries",
                        "target_mass_raw_all_queries",
                        "target_mass_raw_target_position_queries",
                        "target_mass_norm_source_target_queries",
                        "target_mass_raw_source_target_queries",
                        "distractor_mass_norm_all_queries",
                        "target_minus_distractor_mass",
                    ]
                    if case.get("target_box_xyxy") is None:
                        add_check(
                            checks,
                            failures,
                            f"{layer_name} summary_no_target_semantics",
                            not layer.get("target_key_token_indices")
                            and not layer.get("target_query_token_indices")
                            and not layer.get("content_shifted_target_query_token_indices")
                            and not layer.get("source_target_query_token_indices")
                            and not layer.get("distractor_key_token_indices")
                            and all(is_nan_number(layer.get(name)) for name in target_metric_names),
                            {name: layer.get(name) for name in target_metric_names},
                        )
                    else:
                        target_box = [int(value) for value in case["target_box_xyxy"]]
                        image_size = case.get("image_size")
                        target_keys = [int(value) for value in layer.get("target_key_token_indices", [])]
                        target_queries = [int(value) for value in layer.get("target_query_token_indices", [])]
                        shifted_queries = [
                            int(value) for value in layer.get("content_shifted_target_query_token_indices", [])
                        ]
                        source_target_queries = [
                            int(value) for value in layer.get("source_target_query_token_indices", [])
                        ]
                        distractor_keys = [int(value) for value in layer.get("distractor_key_token_indices", [])]
                        expected_target_keys = (
                            bbox_to_token_indices(
                                image_size=image_size,
                                grid=payload["image1_grid"],
                                bbox_xyxy=target_box,
                            )
                            if image_size
                            else []
                        )
                        expected_target_queries = (
                            bbox_to_token_indices(
                                image_size=image_size,
                                grid=payload["image2_grid"],
                                bbox_xyxy=target_box,
                            )
                            if image_size
                            else []
                        )
                        expected_shifted_queries = (
                            []
                            if is_sequence_roll
                            else content_shifted_bbox_token_indices(
                                image_size=image_size,
                                grid=payload["image2_grid"],
                                bbox_xyxy=target_box,
                                transform_record=record,
                            ) if image_size else []
                        )
                        expected_source_target_queries = (
                            [
                                query_index
                                for query_index, source_index in enumerate(source_indices)
                                if int(source_index) in set(expected_target_keys)
                            ]
                            if is_sequence_roll
                            else []
                        )
                        distractor_box = case.get("distractor_box_xyxy")
                        expected_distractor_keys = (
                            bbox_to_token_indices(
                                image_size=image_size,
                                grid=payload["image1_grid"],
                                bbox_xyxy=[int(value) for value in distractor_box],
                            )
                            if distractor_box and image_size
                            else []
                        )
                        target_detail = {
                            "target_keys": target_keys,
                            "expected_target_keys": expected_target_keys,
                            "target_queries": target_queries,
                            "expected_target_queries": expected_target_queries,
                            "shifted_queries": shifted_queries,
                            "expected_shifted_queries": expected_shifted_queries,
                            "source_target_queries": source_target_queries,
                            "expected_source_target_queries": expected_source_target_queries,
                            "distractor_keys": distractor_keys,
                            "expected_distractor_keys": expected_distractor_keys,
                            "metrics": {name: layer.get(name) for name in target_metric_names},
                        }
                        add_check(
                            checks,
                            failures,
                            f"{layer_name} summary_target_indices",
                            bool(image_size)
                            and bool(expected_target_keys)
                            and bool(expected_target_queries)
                            and target_keys == expected_target_keys
                            and target_queries == expected_target_queries
                            and shifted_queries == expected_shifted_queries
                            and source_target_queries == expected_source_target_queries
                            and distractor_keys == expected_distractor_keys
                            and indices_valid(target_keys, image1_tokens)
                            and indices_valid(target_queries, image2_tokens)
                            and indices_valid(shifted_queries, image2_tokens)
                            and indices_valid(source_target_queries, image2_tokens)
                            and indices_valid(distractor_keys, image1_tokens),
                            target_detail,
                        )
                        add_check(
                            checks,
                            failures,
                            f"{layer_name} summary_target_metric_semantics",
                            is_finite_number(layer.get("target_mass_norm_all_queries"))
                            and is_finite_number(layer.get("target_mass_norm_target_queries"))
                            and is_finite_number(layer.get("target_mass_raw_all_queries"))
                            and is_finite_number(layer.get("target_mass_raw_target_position_queries"))
                            and (
                                is_nan_number(layer.get("target_mass_norm_content_shifted_target_queries"))
                                and (
                                    (
                                        is_finite_number(layer.get("target_mass_norm_source_target_queries"))
                                        and is_finite_number(layer.get("target_mass_raw_source_target_queries"))
                                    )
                                    if expected_source_target_queries
                                    else (
                                        is_nan_number(layer.get("target_mass_norm_source_target_queries"))
                                        and is_nan_number(layer.get("target_mass_raw_source_target_queries"))
                                    )
                                )
                                if is_sequence_roll
                                else (
                                    bool(expected_shifted_queries)
                                    and is_finite_number(
                                        layer.get("target_mass_norm_content_shifted_target_queries")
                                    )
                                    and is_nan_number(layer.get("target_mass_norm_source_target_queries"))
                                    and is_nan_number(layer.get("target_mass_raw_source_target_queries"))
                                )
                            )
                            and (
                                is_finite_number(layer.get("distractor_mass_norm_all_queries"))
                                and is_finite_number(layer.get("target_minus_distractor_mass"))
                                if expected_distractor_keys
                                else is_nan_number(layer.get("distractor_mass_norm_all_queries"))
                                and is_nan_number(layer.get("target_minus_distractor_mass"))
                            ),
                            target_detail,
                        )
                    continue
                if dump_mode == "scalar":
                    query_count = int(layer.get("query_count", 0) or 0)
                    image1_key_count = int(layer.get("image1_key_count", 0) or 0)
                    image2_key_count = int(layer.get("image2_key_count", 0) or 0)
                    text_key_count = int(layer.get("text_key_count", 0) or 0)
                    scalar_detail = {
                        "query_count": query_count,
                        "image1_key_count": image1_key_count,
                        "text_key_count": text_key_count,
                        "image2_key_count": image2_key_count,
                        "mean_image1_mass_raw": layer.get("mean_image1_mass_raw"),
                        "mean_text_mass_raw": layer.get("mean_text_mass_raw"),
                        "mean_image2_mass_raw": layer.get("mean_image2_mass_raw"),
                        "mass_total_mean": layer.get("mass_total_mean"),
                        "mass_total_max": layer.get("mass_total_max"),
                        "scalar_npz_path": layer.get("scalar_npz_path", ""),
                    }
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} scalar_counts",
                        query_count == image2_tokens
                        and image1_key_count == image1_tokens
                        and image2_key_count == image2_tokens
                        and text_key_count >= 0,
                        scalar_detail,
                    )
                    scalar_finite = (
                        is_finite_number(layer.get("mean_image1_mass_raw"))
                        and is_finite_number(layer.get("mean_text_mass_raw"))
                        and is_finite_number(layer.get("mean_image2_mass_raw"))
                        and is_finite_number(layer.get("mass_total_mean"))
                        and is_finite_number(layer.get("mass_total_max"))
                    )
                    add_check(checks, failures, f"{layer_name} scalar_finite", scalar_finite, scalar_detail)
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} scalar_mass_leq_one",
                        float(layer.get("mass_total_max", 9.0)) <= 1.0 + row_sum_atol,
                        scalar_detail,
                    )
                    scalar_npz_rel = str(layer.get("scalar_npz_path", "") or "")
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} scalar_raw_present",
                        bool(scalar_npz_rel) or not require_scalar_raw,
                        scalar_detail,
                    )
                    if scalar_npz_rel:
                        scalar_npz_path = output_dir / scalar_npz_rel
                        scalar_data = np.load(scalar_npz_path)
                        image1_mass = np.asarray(scalar_data["image1_mass_raw"], dtype=np.float32)
                        text_mass = np.asarray(scalar_data["text_mass_raw"], dtype=np.float32)
                        image2_mass = np.asarray(scalar_data["image2_mass_raw"], dtype=np.float32)
                        query_rows = np.asarray(scalar_data["query_rows"], dtype=np.int32)
                        query_cols = np.asarray(scalar_data["query_cols"], dtype=np.int32)
                        raw_total = image1_mass + text_mass + image2_mass
                        raw_detail = {
                            **scalar_detail,
                            "image1_mass_shape": list(image1_mass.shape),
                            "text_mass_shape": list(text_mass.shape),
                            "image2_mass_shape": list(image2_mass.shape),
                            "query_rows_shape": list(query_rows.shape),
                            "query_cols_shape": list(query_cols.shape),
                            "raw_total_max": float(raw_total.max()) if raw_total.size else None,
                        }
                        add_check(
                            checks,
                            failures,
                            f"{layer_name} scalar_raw_shapes",
                            image1_mass.shape == (image2_tokens,)
                            and text_mass.shape == (image2_tokens,)
                            and image2_mass.shape == (image2_tokens,)
                            and query_rows.shape == (image2_tokens,)
                            and query_cols.shape == (image2_tokens,),
                            raw_detail,
                        )
                        add_check(
                            checks,
                            failures,
                            f"{layer_name} scalar_raw_finite",
                            bool(
                                np.isfinite(image1_mass).all()
                                and np.isfinite(text_mass).all()
                                and np.isfinite(image2_mass).all()
                            ),
                            raw_detail,
                        )
                        add_check(
                            checks,
                            failures,
                            f"{layer_name} scalar_raw_metrics_match",
                            scalar_mean_matches(layer.get("mean_image1_mass_raw"), image1_mass, metric_atol)
                            and scalar_mean_matches(layer.get("mean_text_mass_raw"), text_mass, metric_atol)
                            and scalar_mean_matches(layer.get("mean_image2_mass_raw"), image2_mass, metric_atol)
                            and metric_matches(layer.get("mass_total_max"), float(raw_total.max()), metric_atol),
                            raw_detail,
                        )
                    if require_target_box:
                        add_check(
                            checks,
                            failures,
                            f"{layer_name} target_box_requires_full_dump",
                            False,
                            "scalar mode does not preserve I2xI1 matrices needed for target-box metrics",
                        )
                    continue

                full_layer_count += 1
                npz_rel = str(layer["npz_path"])
                add_check(
                    checks,
                    failures,
                    f"{layer_name} unique_matrix_path",
                    bool(npz_rel) and npz_rel not in seen_attention_raw_paths,
                    npz_rel,
                )
                seen_attention_raw_paths.add(npz_rel)
                npz_path = output_dir / npz_rel
                data = np.load(npz_path)
                matrix_norm = np.asarray(data["matrix_norm"], dtype=np.float32)
                matrix_raw = np.asarray(data["matrix_raw"], dtype=np.float32)
                image2_block = np.asarray(data["image2_block_raw"], dtype=np.float32)
                image1_mass = np.asarray(data["image1_mass_raw"], dtype=np.float32)
                text_mass = np.asarray(data["text_mass_raw"], dtype=np.float32)
                image2_mass = np.asarray(data["image2_mass_raw"], dtype=np.float32)
                token_ids = np.arange(image2_block.shape[0])
                future_mask = token_ids[None, :] > token_ids[:, None]
                row_error = float(np.max(np.abs(matrix_norm.sum(axis=-1) - 1.0)))
                raw_mass_error = float(np.max(np.abs(matrix_raw.sum(axis=-1) - image1_mass)))
                image2_mass_error = float(np.max(np.abs(image2_block.sum(axis=-1) - image2_mass)))
                future_mass_max = (
                    float(np.max(np.where(future_mask, image2_block, 0.0).sum(axis=-1)))
                    if image2_block.shape[0]
                    else 0.0
                )
                mass_total_max = float(np.max(image1_mass + text_mass + image2_mass))
                mass_total_mean = float(np.mean(image1_mass + text_mass + image2_mass))
                query_rows = np.asarray(data["query_rows"], dtype=np.int64)
                query_cols = np.asarray(data["query_cols"], dtype=np.int64)
                key_rows = np.asarray(data["key_rows"], dtype=np.int64)
                key_cols = np.asarray(data["key_cols"], dtype=np.int64)
                position_cheb = np.maximum(
                    np.abs(query_rows[:, None] - key_rows[None, :]),
                    np.abs(query_cols[:, None] - key_cols[None, :]),
                )
                position_euclid = np.sqrt(
                    (query_rows[:, None] - key_rows[None, :]) ** 2
                    + (query_cols[:, None] - key_cols[None, :]) ** 2
                )
                exact_position = correspondence_band_mass(matrix_norm, position_cheb, 0)
                position_band = correspondence_band_mass(
                    matrix_norm,
                    position_cheb,
                    int(summary.get("band_radius", 1)),
                )
                token_index = np.arange(image2_block.shape[0])
                past_mask = token_index[None, :] < token_index[:, None]
                diagonal_mask = token_index[None, :] == token_index[:, None]
                self_cheb = np.maximum(
                    np.abs(query_rows[:, None] - query_rows[None, :]),
                    np.abs(query_cols[:, None] - query_cols[None, :]),
                )
                local_past_mask = past_mask & (self_cheb <= int(summary.get("band_radius", 1)))
                i2_total_self = float(image2_block.sum(axis=-1).mean())
                i2_past_self = float(image2_block[past_mask].sum() / max(image2_block.shape[0], 1))
                i2_diag_self = float(image2_block[diagonal_mask].sum() / max(image2_block.shape[0], 1))
                i2_local_self = float(image2_block[local_past_mask].sum() / max(image2_block.shape[0], 1))
                i2_local_ratio = float(i2_local_self / max(i2_past_self, 1e-8))
                add_check(
                    checks,
                    failures,
                    f"{layer_name} matrix_shape",
                    matrix_norm.shape == (image2_tokens, image1_tokens),
                    matrix_norm.shape,
                )
                add_check(
                    checks,
                    failures,
                    f"{layer_name} image2_block_shape",
                    image2_block.shape == (image2_tokens, image2_tokens),
                    image2_block.shape,
                )
                add_check(checks, failures, f"{layer_name} row_norm", row_error < row_sum_atol, row_error)
                add_check(checks, failures, f"{layer_name} raw_mass", raw_mass_error < row_sum_atol, raw_mass_error)
                add_check(checks, failures, f"{layer_name} image2_mass", image2_mass_error < row_sum_atol, image2_mass_error)
                add_check(checks, failures, f"{layer_name} future_i2_mass", future_mass_max < row_sum_atol, future_mass_max)
                add_check(
                    checks,
                    failures,
                    f"{layer_name} future_i2_mass_summary_match",
                    metric_matches(layer.get("future_i2_mass_max"), future_mass_max, metric_atol),
                    {"summary": layer.get("future_i2_mass_max"), "raw": future_mass_max},
                )
                add_check(checks, failures, f"{layer_name} mass_leq_one", mass_total_max <= 1.0 + row_sum_atol, mass_total_max)
                finite = (
                    np.isfinite(matrix_norm).all()
                    and np.isfinite(matrix_raw).all()
                    and np.isfinite(image2_block).all()
                    and np.isfinite(image1_mass).all()
                )
                add_check(checks, failures, f"{layer_name} finite", bool(finite), str(npz_path))
                paper_metric_detail = {
                    "mean_image1_mass_raw": float(image1_mass.mean()),
                    "mean_text_mass_raw": float(text_mass.mean()),
                    "mean_image2_mass_raw": float(image2_mass.mean()),
                    "mass_total_mean": mass_total_mean,
                    "mass_total_max": mass_total_max,
                    "position_band_mass": position_band,
                    "exact_position_mass": exact_position,
                    "expected_position_distance": expected_normalized_distance(matrix_norm, position_euclid),
                    "row_entropy": normalized_row_entropy(matrix_norm),
                    "i2_total_self_mass_raw": i2_total_self,
                    "i2_past_self_mass_raw": i2_past_self,
                    "i2_diag_self_mass_raw": i2_diag_self,
                    "i2_local_self_mass_raw": i2_local_self,
                    "i2_local_self_ratio": i2_local_ratio,
                }
                add_check(
                    checks,
                    failures,
                    f"{layer_name} paper_metrics_recomputed",
                    all(
                        metric_matches(layer.get(name), recomputed, metric_atol)
                        for name, recomputed in paper_metric_detail.items()
                    )
                    and metric_matches(layer.get("local_correspondence_band_mass"), position_band, metric_atol)
                    and metric_matches(
                        layer.get("expected_distance_from_diagonal"),
                        paper_metric_detail["expected_position_distance"],
                        metric_atol,
                    ),
                    paper_metric_detail,
                )
                if is_hook_noop:
                    baseline_layers = {
                        int(item["layer"]): item for item in transforms["baseline"]["layers"]
                    }
                    baseline_layer = baseline_layers.get(int(layer["layer"]))
                    baseline_npz = (
                        np.load(output_dir / baseline_layer["npz_path"])
                        if baseline_layer and baseline_layer.get("npz_path")
                        else None
                    )
                    compared_arrays = [
                        "matrix_raw",
                        "matrix_norm",
                        "image2_block_raw",
                        "image1_mass_raw",
                        "text_mass_raw",
                        "image2_mass_raw",
                    ]
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} unhooked_vs_hook_noop_raw_exact",
                        baseline_npz is not None
                        and all(np.array_equal(data[name], baseline_npz[name]) for name in compared_arrays),
                        {"arrays": compared_arrays},
                    )
                if is_sequence_roll:
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} sequence_spatial_content_metrics_nan",
                        is_nan_number(layer.get("content_band_mass"))
                        and is_nan_number(layer.get("expected_content_distance"))
                        and not layer.get("content_distance_profile"),
                        {
                            "content_band_mass": layer.get("content_band_mass"),
                            "expected_content_distance": layer.get("expected_content_distance"),
                            "content_distance_profile": layer.get("content_distance_profile"),
                        },
                    )
                    source = np.asarray(source_indices, dtype=np.int64)
                    source_rows = key_rows[source]
                    source_cols = key_cols[source]
                    cheb = np.maximum(
                        np.abs(source_rows[:, None] - key_rows[None, :]),
                        np.abs(source_cols[:, None] - key_cols[None, :]),
                    )
                    euclid = np.sqrt(
                        (source_rows[:, None] - key_rows[None, :]) ** 2
                        + (source_cols[:, None] - key_cols[None, :]) ** 2
                    )
                    exact_source = correspondence_band_mass(matrix_norm, cheb, 0)
                    source_detail = {
                        "source_index_band_mass": layer.get("source_index_band_mass"),
                        "source_index_exact_mass": layer.get("source_index_exact_mass"),
                        "expected_source_index_distance": layer.get("expected_source_index_distance"),
                        "source_minus_position_exact_mass": layer.get("source_minus_position_exact_mass"),
                    }
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} source_index_metrics_recomputed",
                        metric_matches(
                            layer.get("source_index_band_mass"),
                            correspondence_band_mass(matrix_norm, cheb, int(summary.get("band_radius", 1))),
                            metric_atol,
                        )
                        and metric_matches(layer.get("source_index_exact_mass"), exact_source, metric_atol)
                        and metric_matches(
                            layer.get("expected_source_index_distance"),
                            expected_normalized_distance(matrix_norm, euclid),
                            metric_atol,
                        )
                        and metric_matches(
                            layer.get("source_minus_position_exact_mass"),
                            exact_source - exact_position,
                            metric_atol,
                        ),
                        source_detail,
                    )
                else:
                    content_cheb, content_euclid = spatial_content_distances(
                        query_rows=query_rows,
                        query_cols=query_cols,
                        key_rows=key_rows,
                        key_cols=key_cols,
                        grid_h=int(payload["image1_grid"]["llm_grid_h"]),
                        grid_w=int(payload["image1_grid"]["llm_grid_w"]),
                        transform_record=record,
                    )
                    content_detail = {
                        "content_band_mass": correspondence_band_mass(
                            matrix_norm,
                            content_cheb,
                            int(summary.get("band_radius", 1)),
                        ),
                        "expected_content_distance": expected_normalized_distance(matrix_norm, content_euclid),
                    }
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} content_metrics_recomputed",
                        all(
                            metric_matches(layer.get(name), recomputed, metric_atol)
                            for name, recomputed in content_detail.items()
                        ),
                        content_detail,
                    )
                if require_target_box or case.get("target_box_xyxy"):
                    target_keys = layer.get("target_key_token_indices", [])
                    target_queries = layer.get("target_query_token_indices", [])
                    shifted_queries = layer.get("content_shifted_target_query_token_indices", [])
                    source_target_queries = layer.get("source_target_query_token_indices", [])
                    distractor_keys = layer.get("distractor_key_token_indices", [])
                    target_box = case.get("target_box_xyxy")
                    image_size = case.get("image_size")
                    expected_target_keys: list[int] = []
                    expected_target_queries: list[int] = []
                    expected_shifted_queries: list[int] = []
                    expected_source_target_queries: list[int] = []
                    expected_distractor_keys: list[int] = []
                    if target_box and image_size:
                        expected_target_keys = bbox_to_token_indices(
                            image_size=image_size,
                            grid=payload["image1_grid"],
                            bbox_xyxy=[int(v) for v in target_box],
                        )
                        expected_target_queries = bbox_to_token_indices(
                            image_size=image_size,
                            grid=payload["image2_grid"],
                            bbox_xyxy=[int(v) for v in target_box],
                        )
                        if not is_sequence_roll:
                            expected_shifted_queries = content_shifted_bbox_token_indices(
                                image_size=image_size,
                                grid=payload["image2_grid"],
                                bbox_xyxy=[int(v) for v in target_box],
                                transform_record=record,
                            )
                        else:
                            target_key_set = set(expected_target_keys)
                            expected_source_target_queries = [
                                query_index
                                for query_index, source_index in enumerate(source_indices)
                                if int(source_index) in target_key_set
                            ]
                        if case.get("distractor_box_xyxy"):
                            expected_distractor_keys = bbox_to_token_indices(
                                image_size=image_size,
                                grid=payload["image1_grid"],
                                bbox_xyxy=[int(v) for v in case["distractor_box_xyxy"]],
                            )
                    recomputed_all = mean_target_mass(matrix_norm, None, [int(v) for v in target_keys])
                    recomputed_target = mean_target_mass(
                        matrix_norm, [int(v) for v in target_queries], [int(v) for v in target_keys]
                    )
                    recomputed_shifted = mean_target_mass(
                        matrix_norm, [int(v) for v in shifted_queries], [int(v) for v in target_keys]
                    )
                    recomputed_raw_all = mean_target_mass(matrix_raw, None, [int(v) for v in target_keys])
                    recomputed_raw_target = mean_target_mass(
                        matrix_raw, [int(v) for v in target_queries], [int(v) for v in target_keys]
                    )
                    recomputed_source_norm = mean_target_mass(
                        matrix_norm,
                        [int(v) for v in source_target_queries],
                        [int(v) for v in target_keys],
                    )
                    recomputed_source_raw = mean_target_mass(
                        matrix_raw,
                        [int(v) for v in source_target_queries],
                        [int(v) for v in target_keys],
                    )
                    recomputed_distractor = mean_target_mass(
                        matrix_norm,
                        None,
                        [int(v) for v in distractor_keys],
                    )
                    target_detail = {
                        "target_key_count": len(target_keys),
                        "target_query_count": len(target_queries),
                        "content_shifted_target_query_count": len(shifted_queries),
                        "expected_target_key_count": len(expected_target_keys),
                        "expected_target_query_count": len(expected_target_queries),
                        "expected_content_shifted_target_query_count": len(expected_shifted_queries),
                        "source_target_query_count": len(source_target_queries),
                        "expected_source_target_query_count": len(expected_source_target_queries),
                        "distractor_key_count": len(distractor_keys),
                        "expected_distractor_key_count": len(expected_distractor_keys),
                        "target_mass_norm_all_queries": layer.get("target_mass_norm_all_queries"),
                        "target_mass_norm_target_queries": layer.get("target_mass_norm_target_queries"),
                        "target_mass_norm_content_shifted_target_queries": layer.get(
                            "target_mass_norm_content_shifted_target_queries"
                        ),
                        "recomputed_target_mass_norm_all_queries": recomputed_all,
                        "recomputed_target_mass_norm_target_queries": recomputed_target,
                        "recomputed_target_mass_norm_content_shifted_target_queries": recomputed_shifted,
                        "recomputed_target_mass_raw_all_queries": recomputed_raw_all,
                        "recomputed_target_mass_raw_target_position_queries": recomputed_raw_target,
                        "recomputed_target_mass_norm_source_target_queries": recomputed_source_norm,
                        "recomputed_target_mass_raw_source_target_queries": recomputed_source_raw,
                        "recomputed_distractor_mass_norm_all_queries": recomputed_distractor,
                    }
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} target_box_present",
                        bool(target_box) and bool(image_size),
                        {"target_box_xyxy": target_box, "image_size": image_size},
                    )
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} target_key_nonempty",
                        len(target_keys) > 0,
                        target_detail,
                    )
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} target_query_nonempty",
                        len(target_queries) > 0,
                        target_detail,
                    )
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} content_shifted_target_query_semantics",
                        (not is_sequence_roll and len(shifted_queries) > 0)
                        or (is_sequence_roll and len(shifted_queries) == 0),
                        target_detail,
                    )
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} target_metrics_finite",
                        is_finite_number(layer.get("target_mass_norm_all_queries"))
                        and is_finite_number(layer.get("target_mass_norm_target_queries"))
                        and (
                            is_finite_number(layer.get("target_mass_norm_content_shifted_target_queries"))
                            if not is_sequence_roll
                            else is_nan_number(layer.get("target_mass_norm_content_shifted_target_queries"))
                        ),
                        target_detail,
                    )
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} target_indices_in_range_unique",
                        indices_valid([int(v) for v in target_keys], matrix_norm.shape[1])
                        and indices_valid([int(v) for v in target_queries], matrix_norm.shape[0])
                        and indices_valid([int(v) for v in shifted_queries], matrix_norm.shape[0])
                        and indices_valid([int(v) for v in source_target_queries], matrix_norm.shape[0])
                        and indices_valid([int(v) for v in distractor_keys], matrix_norm.shape[1]),
                        target_detail,
                    )
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} target_indices_recomputed",
                        [int(v) for v in target_keys] == expected_target_keys
                        and [int(v) for v in target_queries] == expected_target_queries
                        and [int(v) for v in shifted_queries] == expected_shifted_queries
                        and [int(v) for v in source_target_queries] == expected_source_target_queries
                        and [int(v) for v in distractor_keys] == expected_distractor_keys,
                        target_detail,
                    )
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} target_raw_source_metrics_match_npz",
                        metric_matches(layer.get("target_mass_raw_all_queries"), recomputed_raw_all, metric_atol)
                        and metric_matches(
                            layer.get("target_mass_raw_target_position_queries"),
                            recomputed_raw_target,
                            metric_atol,
                        )
                        and (
                            metric_matches(
                                layer.get("target_mass_norm_source_target_queries"),
                                recomputed_source_norm,
                                metric_atol,
                            )
                            and metric_matches(
                                layer.get("target_mass_raw_source_target_queries"),
                                recomputed_source_raw,
                                metric_atol,
                            )
                            if expected_source_target_queries
                            else is_nan_number(layer.get("target_mass_norm_source_target_queries"))
                            and is_nan_number(layer.get("target_mass_raw_source_target_queries"))
                        )
                        and (
                            metric_matches(
                                layer.get("distractor_mass_norm_all_queries"),
                                recomputed_distractor,
                                metric_atol,
                            )
                            and metric_matches(
                                layer.get("target_minus_distractor_mass"),
                                recomputed_all - recomputed_distractor,
                                metric_atol,
                            )
                            if expected_distractor_keys
                            else is_nan_number(layer.get("distractor_mass_norm_all_queries"))
                            and is_nan_number(layer.get("target_minus_distractor_mass"))
                        ),
                        target_detail,
                    )
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} target_metrics_match_npz",
                        metric_matches(layer.get("target_mass_norm_all_queries"), recomputed_all, metric_atol)
                        and metric_matches(layer.get("target_mass_norm_target_queries"), recomputed_target, metric_atol)
                        and (
                            metric_matches(
                                layer.get("target_mass_norm_content_shifted_target_queries"),
                                recomputed_shifted,
                                metric_atol,
                            )
                            if not is_sequence_roll
                            else is_nan_number(layer.get("target_mass_norm_content_shifted_target_queries"))
                        ),
                        target_detail,
                    )
                else:
                    no_target_metric_names = [
                        "target_mass_norm_all_queries",
                        "target_mass_norm_target_queries",
                        "target_mass_norm_content_shifted_target_queries",
                        "target_mass_raw_all_queries",
                        "target_mass_raw_target_position_queries",
                        "target_mass_norm_source_target_queries",
                        "target_mass_raw_source_target_queries",
                        "distractor_mass_norm_all_queries",
                        "target_minus_distractor_mass",
                    ]
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} full_no_target_semantics",
                        not layer.get("target_key_token_indices")
                        and not layer.get("target_query_token_indices")
                        and not layer.get("content_shifted_target_query_token_indices")
                        and not layer.get("source_target_query_token_indices")
                        and not layer.get("distractor_key_token_indices")
                        and all(is_nan_number(layer.get(name)) for name in no_target_metric_names),
                        {name: layer.get(name) for name in no_target_metric_names},
                    )

    if require_visual_sequence_raw:
        add_check(
            checks,
            failures,
            "visual_sequence_raw_present",
            raw_sequence_count > 0,
            raw_sequence_count,
        )
    if require_processor_pair_raw:
        add_check(
            checks,
            failures,
            "processor_pair_raw_present",
            raw_processor_pair_count > 0,
            raw_processor_pair_count,
        )
    if strict_logicvista100:
        expected_cells_per_case = len(expected_transform_set) * len(selected_layer_ids)
        expected_full_layers = expected_cells_per_case
        expected_summary_layers = max(0, len(summary.get("cases", [])) - 1) * expected_cells_per_case
        actual_npz_paths = {
            str(path.relative_to(output_dir))
            for path in (output_dir / "npz").rglob("*.npz")
        }
        add_check(
            checks,
            failures,
            "logicvista100_bounded_full_raw_layers",
            full_layer_count == expected_full_layers and summary_layer_count == expected_summary_layers,
            {
                "full_layer_count": full_layer_count,
                "expected_full_layers": expected_full_layers,
                "summary_layer_count": summary_layer_count,
                "expected_summary_layers": expected_summary_layers,
            },
        )
        add_check(
            checks,
            failures,
            "logicvista100_exact_matrix_files",
            actual_npz_paths == seen_attention_raw_paths
            and len(actual_npz_paths) == expected_full_layers,
            {
                "actual_count": len(actual_npz_paths),
                "referenced_count": len(seen_attention_raw_paths),
                "expected_count": expected_full_layers,
                "extra": sorted(actual_npz_paths - seen_attention_raw_paths),
                "missing": sorted(seen_attention_raw_paths - actual_npz_paths),
            },
        )

    return {
        "ok": not failures,
        "check_count": len(checks),
        "failure_count": len(failures),
        "row_sum_atol": float(row_sum_atol),
        "metric_atol": float(metric_atol),
        "failures": failures,
        "checks_preview": checks[:40],
    }


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    result = validate(
        output_dir,
        expected_cases=args.expected_cases,
        row_sum_atol=args.row_sum_atol,
        metric_atol=args.metric_atol,
        require_target_box=args.require_target_box,
        require_scalar_raw=args.require_scalar_raw,
        expected_interventions=args.expected_interventions,
        strict_contract=args.strict_contract,
        strict_logicvista100=args.strict_logicvista100,
        require_visual_sequence_raw=args.require_visual_sequence_raw,
        require_processor_pair_raw=args.require_processor_pair_raw,
    )
    (output_dir / "smoke_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
