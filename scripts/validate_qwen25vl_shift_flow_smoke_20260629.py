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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate raw smoke artifacts from qwen25vl_shift_flow_probe.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-cases", type=int, default=0)
    parser.add_argument("--row-sum-atol", type=float, default=2e-3)
    parser.add_argument("--metric-atol", type=float, default=5e-3)
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

    if strict_contract:
        add_check(checks, failures, "strict_expected_cases", expected_cases > 0, expected_cases)
        add_check(
            checks,
            failures,
            "strict_expected_interventions",
            expected_interventions is not None and len(expected_interventions) > 0,
            expected_interventions,
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
            summary.get("dump_mode") == "full" and summary.get("band_radius") == 1,
            {"dump_mode": summary.get("dump_mode"), "band_radius": summary.get("band_radius")},
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

    for case in summary["cases"]:
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
            for layer in payload["layers"]:
                layer_name = f"{case['case_id']} {transform} L{layer['layer']}"
                dump_mode = str(layer.get("dump_mode", "full"))
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

                npz_path = output_dir / layer["npz_path"]
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
                add_check(checks, failures, f"{layer_name} mass_leq_one", mass_total_max <= 1.0 + row_sum_atol, mass_total_max)
                finite = (
                    np.isfinite(matrix_norm).all()
                    and np.isfinite(matrix_raw).all()
                    and np.isfinite(image2_block).all()
                    and np.isfinite(image1_mass).all()
                )
                add_check(checks, failures, f"{layer_name} finite", bool(finite), str(npz_path))
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
                    key_rows = np.asarray(data["key_rows"], dtype=np.int64)
                    key_cols = np.asarray(data["key_cols"], dtype=np.int64)
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
                    exact_position = float(np.diag(matrix_norm).sum() / max(matrix_norm.shape[0], 1))
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
                if require_target_box or case.get("target_box_xyxy"):
                    target_keys = layer.get("target_key_token_indices", [])
                    target_queries = layer.get("target_query_token_indices", [])
                    shifted_queries = layer.get("content_shifted_target_query_token_indices", [])
                    target_box = case.get("target_box_xyxy")
                    image_size = case.get("image_size")
                    expected_target_keys: list[int] = []
                    expected_target_queries: list[int] = []
                    expected_shifted_queries: list[int] = []
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
                    recomputed_all = mean_target_mass(matrix_norm, None, [int(v) for v in target_keys])
                    recomputed_target = mean_target_mass(
                        matrix_norm, [int(v) for v in target_queries], [int(v) for v in target_keys]
                    )
                    recomputed_shifted = mean_target_mass(
                        matrix_norm, [int(v) for v in shifted_queries], [int(v) for v in target_keys]
                    )
                    target_detail = {
                        "target_key_count": len(target_keys),
                        "target_query_count": len(target_queries),
                        "content_shifted_target_query_count": len(shifted_queries),
                        "expected_target_key_count": len(expected_target_keys),
                        "expected_target_query_count": len(expected_target_queries),
                        "expected_content_shifted_target_query_count": len(expected_shifted_queries),
                        "target_mass_norm_all_queries": layer.get("target_mass_norm_all_queries"),
                        "target_mass_norm_target_queries": layer.get("target_mass_norm_target_queries"),
                        "target_mass_norm_content_shifted_target_queries": layer.get(
                            "target_mass_norm_content_shifted_target_queries"
                        ),
                        "recomputed_target_mass_norm_all_queries": recomputed_all,
                        "recomputed_target_mass_norm_target_queries": recomputed_target,
                        "recomputed_target_mass_norm_content_shifted_target_queries": recomputed_shifted,
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
                        and indices_valid([int(v) for v in shifted_queries], matrix_norm.shape[0]),
                        target_detail,
                    )
                    add_check(
                        checks,
                        failures,
                        f"{layer_name} target_indices_recomputed",
                        [int(v) for v in target_keys] == expected_target_keys
                        and [int(v) for v in target_queries] == expected_target_queries
                        and [int(v) for v in shifted_queries] == expected_shifted_queries,
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

    return {
        "ok": not failures,
        "check_count": len(checks),
        "failure_count": len(failures),
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
