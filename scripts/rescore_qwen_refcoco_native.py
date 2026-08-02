#!/usr/bin/env python3
"""Rescore Qwen2.5-VL RefCOCO predictions in the model's native coordinates.

Qwen2.5-VL may emit either the prompt-requested normalized coordinates or its
native absolute coordinates on the smart-resized image. This scorer supports
exactly those two declared conventions without consulting ground truth to pick
an interpretation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
DEFAULT_FACTOR = 28
DEFAULT_MIN_PIXELS = 1280 * 28 * 28
DEFAULT_MAX_PIXELS = 16384 * 28 * 28
DEFAULT_MAX_RATIO = 200.0
IOU_THRESHOLD = 0.5
EXPECTED_COLUMNS = {
    "index",
    "split",
    "bbox_x1_norm",
    "bbox_y1_norm",
    "bbox_x2_norm",
    "bbox_y2_norm",
    "prediction",
}
GT_COLUMNS = ["bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm"]

_NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_SEPARATOR = r"(?:\s*,\s*|\s+)"
_DIRECT_BOX_RE = re.compile(
    rf"\[\s*({_NUMBER}){_SEPARATOR}({_NUMBER}){_SEPARATOR}"
    rf"({_NUMBER}){_SEPARATOR}({_NUMBER})\s*\]"
)
_BBOX_KEY_RE = re.compile(
    rf"(?:[\"']?bbox_2d[\"']?)\s*:\s*\[\s*"
    rf"({_NUMBER}){_SEPARATOR}({_NUMBER}){_SEPARATOR}"
    rf"({_NUMBER}){_SEPARATOR}({_NUMBER})"
    rf"(?:\s*(?P<box_close>\])|\s*(?P<open_end>$))",
    flags=re.IGNORECASE,
)
_FENCED_RE = re.compile(r"\s*```(?:json)?\s*(.*?)\s*```\s*", flags=re.DOTALL | re.IGNORECASE)
_SAFE_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_complete_box_parse(parse_method: str) -> bool:
    """Return whether the parser observed a closing bracket for the box."""
    return parse_method != "bbox_2d_open_array_recovery"


def load_dimension_records(path: Path) -> dict[str, tuple[int, int]]:
    dimensions: dict[str, tuple[int, int]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split("\t")
        if len(parts) != 4:
            raise ValueError(f"{path}:{line_number}: expected four TSV fields.")
        index, _image_path, file_size_text, geometry = parts
        if not index or index in dimensions:
            raise ValueError(f"{path}:{line_number}: empty or duplicate index {index!r}.")
        try:
            file_size = int(file_size_text)
            width_text, height_text = geometry.split("x", 1)
            width = int(width_text)
            height = int(height_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid size or geometry.") from exc
        if file_size <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"{path}:{line_number}: non-positive file metadata.")
        dimensions[index] = (width, height)
    return dimensions


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = DEFAULT_FACTOR,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_ratio: float = DEFAULT_MAX_RATIO,
) -> tuple[int, int]:
    """Match qwen-vl-utils 0.0.11 smart_resize exactly."""
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}.")
    if factor <= 0 or min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("Invalid smart-resize contract.")
    ratio = max(height, width) / min(height, width)
    if ratio > max_ratio:
        raise ValueError(f"Absolute aspect ratio must be <= {max_ratio}, got {ratio}.")

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor((height / beta) / factor) * factor
        w_bar = math.floor((width / beta) / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil((height * beta) / factor) * factor
        w_bar = math.ceil((width * beta) / factor) * factor
    return int(h_bar), int(w_bar)


def _as_box(value: Any) -> Optional[np.ndarray]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    coords = np.asarray(value, dtype=float)
    if coords.shape != (4,) or not np.isfinite(coords).all():
        return None
    return coords


def _extract_single_json_box(value: Any) -> Optional[np.ndarray]:
    direct = _as_box(value)
    if direct is not None:
        return direct
    if isinstance(value, dict):
        candidate = _as_box(value.get("bbox_2d"))
        if candidate is None:
            return None
        other_box_like_values = [
            item
            for key, item in value.items()
            if key != "bbox_2d" and _as_box(item) is not None
        ]
        return None if other_box_like_values else candidate
    if isinstance(value, list) and len(value) == 1:
        return _extract_single_json_box(value[0])
    return None


def parse_qwen_box(text: object) -> tuple[Optional[np.ndarray], str]:
    """Parse one Qwen box while rejecting prose and ambiguous multi-box output."""
    if not isinstance(text, str) or not text.strip():
        return None, "unparsed"
    stripped = text.strip()
    key_occurrences = len(re.findall(r"\bbbox_2d\b", stripped, flags=re.IGNORECASE))
    if key_occurrences > 1:
        return None, "unparsed"
    fenced = _FENCED_RE.fullmatch(stripped)
    json_text = fenced.group(1).strip() if fenced else stripped
    try:
        parsed = json.loads(json_text)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if parsed is not None:
        coords = _extract_single_json_box(parsed)
        if coords is not None:
            return coords, "json_single_box"

    # max_tokens=32 can truncate a Qwen-native JSON answer immediately after
    # the fourth bbox_2d number. The explicit key keeps this recovery narrow.
    keyed_matches = list(_BBOX_KEY_RE.finditer(stripped))
    if len(keyed_matches) == 1 and key_occurrences == 1:
        keyed = keyed_matches[0]
        unrelated_complete_boxes = [
            match
            for match in _DIRECT_BOX_RE.finditer(stripped)
            if match.end() <= keyed.start() or match.start() >= keyed.end() + 1
        ]
        if unrelated_complete_boxes:
            return None, "unparsed"
        coords = np.asarray([float(keyed.group(i)) for i in range(1, 5)], dtype=float)
        method = (
            "bbox_2d_box_complete_recovery"
            if keyed.group("box_close") is not None
            else "bbox_2d_open_array_recovery"
        )
        return coords, method

    direct = _DIRECT_BOX_RE.fullmatch(stripped)
    if direct is not None:
        coords = np.asarray([float(direct.group(i)) for i in range(1, 5)], dtype=float)
        return coords, "direct_box"
    return None, "unparsed"


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x_left = max(float(box1[0]), float(box2[0]))
    y_top = max(float(box1[1]), float(box2[1]))
    x_right = min(float(box1[2]), float(box2[2]))
    y_bottom = min(float(box1[3]), float(box2[3]))
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = max(float(box1[2] - box1[0]), 0.0) * max(float(box1[3] - box1[1]), 0.0)
    area2 = max(float(box2[2] - box2[0]), 0.0) * max(float(box2[3] - box2[1]), 0.0)
    union = area1 + area2 - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def normalize_prediction(
    coords: np.ndarray,
    *,
    source_height: int,
    source_width: int,
    factor: int = DEFAULT_FACTOR,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> tuple[Optional[np.ndarray], str, int, int]:
    resized_height, resized_width = smart_resize(
        source_height,
        source_width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    coords = np.asarray(coords, dtype=float)
    if np.all((coords >= 0.0) & (coords <= 1.0)):
        return coords.copy(), "normalized_0_1", resized_height, resized_width
    integer_valued = np.all(np.isclose(coords, np.rint(coords), rtol=0.0, atol=1e-9))
    if np.any(coords > 1.0) and integer_valued:
        normalized = coords / np.asarray(
            [resized_width, resized_height, resized_width, resized_height],
            dtype=float,
        )
        return normalized, "processed_image_absolute", resized_height, resized_width
    return None, "ambiguous_mixed_protocol", resized_height, resized_width


def score_prediction(
    prediction: object,
    gt_normalized: np.ndarray,
    *,
    source_height: int,
    source_width: int,
    factor: int = DEFAULT_FACTOR,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> dict[str, Any]:
    coords, parse_method = parse_qwen_box(prediction)
    resized_height, resized_width = smart_resize(
        source_height,
        source_width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if coords is None:
        return {
            "parse_method": parse_method,
            "raw_coords": None,
            "coordinate_space": None,
            "resized_width": resized_width,
            "resized_height": resized_height,
            "normalized_coords": None,
            "parsed": False,
            "protocol_valid": False,
            "ordered": False,
            "in_bounds": False,
            "iou": 0.0,
            "hit": False,
        }

    normalized, coordinate_space, _, _ = normalize_prediction(
        coords,
        source_height=source_height,
        source_width=source_width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if normalized is None:
        return {
            "parse_method": parse_method,
            "raw_coords": coords.tolist(),
            "coordinate_space": coordinate_space,
            "resized_width": resized_width,
            "resized_height": resized_height,
            "normalized_coords": None,
            "parsed": True,
            "protocol_valid": False,
            "ordered": False,
            "in_bounds": False,
            "iou": 0.0,
            "hit": False,
        }
    ordered = bool(normalized[2] >= normalized[0] and normalized[3] >= normalized[1])
    in_bounds = bool(np.all((normalized >= 0.0) & (normalized <= 1.0)))
    iou = compute_iou(normalized, gt_normalized) if ordered and in_bounds else 0.0
    return {
        "parse_method": parse_method,
        "raw_coords": coords.tolist(),
        "coordinate_space": coordinate_space,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "normalized_coords": normalized.tolist(),
        "parsed": True,
        "protocol_valid": True,
        "ordered": ordered,
        "in_bounds": in_bounds,
        "iou": iou,
        "hit": bool(iou >= IOU_THRESHOLD),
    }


def _validate_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("Manifest must contain a contract object.")
    expected = {
        "factor": DEFAULT_FACTOR,
        "min_pixels": DEFAULT_MIN_PIXELS,
        "max_pixels": DEFAULT_MAX_PIXELS,
        "expected_rows": 9602,
        "split": "RefCOCOg_test",
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"Manifest contract {key} must be {value!r}, got {contract.get(key)!r}.")
    return contract


def _safe_cell_name(model: object, condition: object) -> str:
    model_text = str(model)
    condition_text = str(condition)
    if _SAFE_NAME_RE.fullmatch(model_text) is None or _SAFE_NAME_RE.fullmatch(condition_text) is None:
        raise ValueError(f"Unsafe cell name: model={model_text!r}, condition={condition_text!r}.")
    return f"{model_text}__{condition_text}"


def _score_cell(
    cell: dict[str, Any],
    *,
    manifest_dir: Path,
    details_dir: Path,
    contract: dict[str, Any],
    dimensions: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    cell_name = _safe_cell_name(cell.get("model"), cell.get("condition"))
    input_path = Path(str(cell.get("input", "")))
    if not input_path.is_absolute():
        input_path = manifest_dir / input_path
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    input_sha256 = sha256_file(input_path)
    expected_input_sha256 = cell.get("input_sha256")
    if input_sha256 != expected_input_sha256:
        raise ValueError(
            f"{cell_name}: input SHA256 mismatch: expected {expected_input_sha256}, "
            f"got {input_sha256}."
        )

    source_manifest_path = Path(str(cell.get("source_prediction_manifest", "")))
    if not source_manifest_path.is_absolute():
        source_manifest_path = manifest_dir / source_manifest_path
    source_manifest_path = source_manifest_path.resolve()
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    source_manifest_sha256 = sha256_file(source_manifest_path)
    expected_source_manifest_sha256 = cell.get("source_prediction_manifest_sha256")
    if source_manifest_sha256 != expected_source_manifest_sha256:
        raise ValueError(
            f"{cell_name}: source prediction manifest SHA256 mismatch: expected "
            f"{expected_source_manifest_sha256}, got {source_manifest_sha256}."
        )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    expected_mode = "image_text" if str(cell["condition"]) == "IQ" else "image_text_image_text"
    source_task = source_manifest.get("task", {})
    source_checks = {
        "artifact_type": source_manifest.get("artifact_type") == "prediction",
        "status": source_manifest.get("status") == "complete",
        "expected_rows": source_manifest.get("expected_rows") == contract["expected_rows"],
        "prediction_sha256": source_manifest.get("prediction_sha256") == input_sha256,
        "model_key": source_task.get("model_key") == str(cell["model"]),
        "mode": source_task.get("mode") == expected_mode,
        "dataset": source_task.get("dataset") == "RefCOCO",
    }
    failed_source_checks = sorted(key for key, passed in source_checks.items() if not passed)
    if failed_source_checks:
        raise ValueError(
            f"{cell_name}: source prediction manifest failed checks {failed_source_checks}."
        )

    data = pd.read_excel(input_path)
    missing_columns = sorted(EXPECTED_COLUMNS - set(data.columns))
    if missing_columns:
        raise ValueError(f"{cell_name}: missing columns {missing_columns}.")
    if len(data) != contract["expected_rows"]:
        raise ValueError(f"{cell_name}: expected {contract['expected_rows']} rows, got {len(data)}.")
    if data["index"].astype(str).duplicated().any():
        raise ValueError(f"{cell_name}: duplicate indices.")
    splits = set(data["split"].astype(str))
    if splits != {contract["split"]}:
        raise ValueError(f"{cell_name}: unexpected splits {sorted(splits)}.")

    parse_methods: Counter[str] = Counter()
    coordinate_spaces: Counter[str] = Counter()
    parsed_count = 0
    protocol_valid_count = 0
    ordered_count = 0
    in_bounds_count = 0
    native_recovered_hits = 0
    native_complete_hits = 0
    iou_sum = 0.0
    resize_sizes: set[tuple[int, int]] = set()
    detail_path = details_dir / f"{cell_name}.jsonl.gz"
    with gzip.open(detail_path, "wt", encoding="utf-8") as handle:
        for row in data.to_dict("records"):
            gt = np.asarray([row[column] for column in GT_COLUMNS], dtype=float)
            if gt.shape != (4,) or not np.isfinite(gt).all() or np.any(gt < 0) or np.any(gt > 1):
                raise ValueError(f"{cell_name}: invalid GT for index {row['index']!r}.")
            index = str(row["index"])
            if index not in dimensions:
                raise ValueError(f"{cell_name}: missing source dimensions for index {index!r}.")
            source_width, source_height = dimensions[index]
            scored = score_prediction(
                row.get("prediction"),
                gt,
                source_height=source_height,
                source_width=source_width,
                factor=contract["factor"],
                min_pixels=contract["min_pixels"],
                max_pixels=contract["max_pixels"],
            )
            parse_methods[scored["parse_method"]] += 1
            if scored["coordinate_space"] is not None:
                coordinate_spaces[scored["coordinate_space"]] += 1
            parsed_count += int(scored["parsed"])
            protocol_valid_count += int(scored["protocol_valid"])
            ordered_count += int(scored["ordered"])
            in_bounds_count += int(scored["in_bounds"])
            native_recovered_hits += int(scored["hit"])
            if is_complete_box_parse(scored["parse_method"]):
                native_complete_hits += int(scored["hit"])
            iou_sum += float(scored["iou"])
            resize_sizes.add((scored["resized_width"], scored["resized_height"]))
            detail = {
                "index": index,
                "prediction": str(row.get("prediction", "")),
                "source_width": source_width,
                "source_height": source_height,
                "gt_normalized": gt.tolist(),
                **scored,
            }
            handle.write(json.dumps(detail, ensure_ascii=True, separators=(",", ":")) + "\n")

    total = len(data)
    return {
        "cell": cell_name,
        "model": str(cell["model"]),
        "condition": str(cell["condition"]),
        "source_task": cell.get("source_task"),
        "input": str(input_path),
        "input_sha256": input_sha256,
        "source_prediction_manifest": str(source_manifest_path),
        "source_prediction_manifest_sha256": source_manifest_sha256,
        "details": str(detail_path.resolve()),
        "details_sha256": sha256_file(detail_path),
        "samples": total,
        "parsed": parsed_count,
        "protocol_valid": protocol_valid_count,
        "ordered": ordered_count,
        "in_bounds": in_bounds_count,
        "hits": native_complete_hits,
        "native_complete_hits": native_complete_hits,
        "native_recovered_hits": native_recovered_hits,
        "parse_success_pct": parsed_count / total * 100.0,
        "protocol_valid_pct": protocol_valid_count / total * 100.0,
        "ordered_box_pct": ordered_count / total * 100.0,
        "in_bounds_pct": in_bounds_count / total * 100.0,
        "precision_at_1": native_complete_hits / total * 100.0,
        "native_recovered_precision_at_1": native_recovered_hits / total * 100.0,
        "native_complete_precision_at_1": native_complete_hits / total * 100.0,
        "average_iou": iou_sum / total,
        "parse_method_counts": dict(sorted(parse_methods.items())),
        "coordinate_space_counts": dict(sorted(coordinate_spaces.items())),
        "unique_resized_sizes": len(resize_sizes),
    }


def run(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = _validate_contract(manifest)
    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("Manifest must contain a non-empty cells list.")
    names = [_safe_cell_name(cell.get("model"), cell.get("condition")) for cell in cells]
    if len(names) != len(set(names)):
        raise ValueError("Manifest contains duplicate model/condition cells.")

    dimension_records_path = Path(str(manifest.get("dimension_records", "")))
    if not dimension_records_path.is_absolute():
        dimension_records_path = manifest_path.parent / dimension_records_path
    dimension_records_path = dimension_records_path.resolve()
    if not dimension_records_path.is_file():
        raise FileNotFoundError(dimension_records_path)
    dimension_records_sha256 = sha256_file(dimension_records_path)
    expected_dimension_sha256 = manifest.get("dimension_records_sha256")
    if dimension_records_sha256 != expected_dimension_sha256:
        raise ValueError(
            "Dimension-record SHA256 mismatch: "
            f"expected {expected_dimension_sha256}, got {dimension_records_sha256}."
        )
    dimensions = load_dimension_records(dimension_records_path)
    if len(dimensions) != contract["expected_rows"]:
        raise ValueError(
            f"Expected {contract['expected_rows']} dimension records, got {len(dimensions)}."
        )

    source_dimension_manifests = []
    declared_dimension_manifests = manifest.get("source_dimension_manifests")
    if not isinstance(declared_dimension_manifests, list) or not declared_dimension_manifests:
        raise ValueError("Manifest must bind at least one source dimension manifest.")
    for declared in declared_dimension_manifests:
        source_path = Path(str(declared.get("path", "")))
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        source_path = source_path.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_sha256 = sha256_file(source_path)
        if source_sha256 != declared.get("sha256"):
            raise ValueError(f"Source dimension manifest SHA256 mismatch: {source_path}.")
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        if source_payload.get("records_sha256") != dimension_records_sha256:
            raise ValueError(f"Source dimension records SHA256 mismatch: {source_path}.")
        if source_payload.get("selected_count") != contract["expected_rows"]:
            raise ValueError(f"Source dimension count mismatch: {source_path}.")
        if source_payload.get("split_value") != contract["split"]:
            raise ValueError(f"Source dimension split mismatch: {source_path}.")
        source_dimension_manifests.append(
            {
                "source_task": declared.get("source_task"),
                "path": str(source_path),
                "sha256": source_sha256,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    details_dir = output_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        _score_cell(
            cell,
            manifest_dir=manifest_path.parent,
            details_dir=details_dir,
            contract=contract,
            dimensions=dimensions,
        )
        for cell in cells
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "dimension_records": str(dimension_records_path),
        "dimension_records_sha256": dimension_records_sha256,
        "dimension_record_count": len(dimensions),
        "source_dimension_manifests": source_dimension_manifests,
        "contract": contract,
        "score_protocol": "qwen_native_aware_dual_coordinate_geometric",
        "cells": summaries,
    }
    summary_path = output_dir / "qwen_refcoco_native_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.manifest, args.output_dir)
    print(json.dumps({"cells": result["cells"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
