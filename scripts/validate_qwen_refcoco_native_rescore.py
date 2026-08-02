#!/usr/bin/env python3
"""Validate Qwen-native RefCOCO rescoring from row-level artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_CELLS = {
    f"qwen25vl_{size}__{condition}"
    for size in ("3b", "7b", "32b", "72b")
    for condition in ("IQ", "IQIQ")
}
EXPECTED_SAMPLES = 9602
IOU_THRESHOLD = 0.5
FACTOR = 28
MIN_PIXELS = 1280 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
GT_COLUMNS = ["bbox_x1_norm", "bbox_y1_norm", "bbox_x2_norm", "bbox_y2_norm"]
EXPECTED_COLUMNS = {"index", "split", "prediction", *GT_COLUMNS}

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dimensions(path: Path) -> dict[str, tuple[int, int]]:
    dimensions: dict[str, tuple[int, int]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split("\t")
        if len(parts) != 4:
            raise ValueError(f"{path}:{line_number}: malformed dimension record.")
        index, _image_path, file_size_text, geometry = parts
        width_text, height_text = geometry.split("x", 1)
        width, height, file_size = int(width_text), int(height_text), int(file_size_text)
        if index in dimensions or min(width, height, file_size) <= 0:
            raise ValueError(f"{path}:{line_number}: invalid dimension record.")
        dimensions[index] = (width, height)
    return dimensions


def independent_iou(box1: list[float], box2: list[float]) -> float:
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = max(box1[2] - box1[0], 0.0) * max(box1[3] - box1[1], 0.0)
    area2 = max(box2[2] - box2[0], 0.0) * max(box2[3] - box2[1], 0.0)
    union = area1 + area2 - intersection
    return 0.0 if union <= 0 else intersection / union


def independent_smart_resize(height: int, width: int) -> tuple[int, int]:
    h_bar = max(FACTOR, round(height / FACTOR) * FACTOR)
    w_bar = max(FACTOR, round(width / FACTOR) * FACTOR)
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        h_bar = math.floor((height / beta) / FACTOR) * FACTOR
        w_bar = math.floor((width / beta) / FACTOR) * FACTOR
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil((height * beta) / FACTOR) * FACTOR
        w_bar = math.ceil((width * beta) / FACTOR) * FACTOR
    return int(h_bar), int(w_bar)


def _independent_as_box(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    coords = [float(item) for item in value]
    return coords if all(math.isfinite(item) for item in coords) else None


def _independent_extract_single_json_box(value: Any) -> list[float] | None:
    direct = _independent_as_box(value)
    if direct is not None:
        return direct
    if isinstance(value, dict):
        candidate = _independent_as_box(value.get("bbox_2d"))
        if candidate is None:
            return None
        other_boxes = [
            item
            for key, item in value.items()
            if key != "bbox_2d" and _independent_as_box(item) is not None
        ]
        return None if other_boxes else candidate
    if isinstance(value, list) and len(value) == 1:
        return _independent_extract_single_json_box(value[0])
    return None


def independent_parse_qwen_box(text: object) -> tuple[list[float] | None, str]:
    if not isinstance(text, str) or not text.strip():
        return None, "unparsed"
    stripped = text.strip()
    key_occurrences = len(re.findall(r"\bbbox_2d\b", stripped, flags=re.IGNORECASE))
    if key_occurrences > 1:
        return None, "unparsed"
    fenced = _FENCED_RE.fullmatch(stripped)
    json_text = fenced.group(1).strip() if fenced else stripped
    try:
        parsed_json = json.loads(json_text)
    except (json.JSONDecodeError, TypeError):
        parsed_json = None
    if parsed_json is not None:
        coords = _independent_extract_single_json_box(parsed_json)
        if coords is not None:
            return coords, "json_single_box"

    keyed_matches = list(_BBOX_KEY_RE.finditer(stripped))
    if len(keyed_matches) == 1 and key_occurrences == 1:
        keyed = keyed_matches[0]
        unrelated_boxes = [
            match
            for match in _DIRECT_BOX_RE.finditer(stripped)
            if match.end() <= keyed.start() or match.start() >= keyed.end() + 1
        ]
        if unrelated_boxes:
            return None, "unparsed"
        coords = [float(keyed.group(i)) for i in range(1, 5)]
        method = (
            "bbox_2d_box_complete_recovery"
            if keyed.group("box_close") is not None
            else "bbox_2d_open_array_recovery"
        )
        return coords, method

    direct = _DIRECT_BOX_RE.fullmatch(stripped)
    if direct is not None:
        return [float(direct.group(i)) for i in range(1, 5)], "direct_box"
    return None, "unparsed"


def independently_recompute_detail(
    source_row: dict[str, Any], source_width: int, source_height: int
) -> dict[str, Any]:
    prediction = str(source_row.get("prediction", ""))
    gt = [float(source_row[column]) for column in GT_COLUMNS]
    coords, parse_method = independent_parse_qwen_box(prediction)
    resized_height, resized_width = independent_smart_resize(source_height, source_width)
    coordinate_space = None
    normalized = None
    parsed = coords is not None
    protocol_valid = ordered = in_bounds = hit = False
    iou = 0.0
    if coords is not None:
        if all(0.0 <= value <= 1.0 for value in coords):
            coordinate_space = "normalized_0_1"
            normalized = list(coords)
        elif any(value > 1.0 for value in coords) and all(
            _close(value, round(value), atol=1e-9) for value in coords
        ):
            coordinate_space = "processed_image_absolute"
            normalized = [
                coords[0] / resized_width,
                coords[1] / resized_height,
                coords[2] / resized_width,
                coords[3] / resized_height,
            ]
        else:
            coordinate_space = "ambiguous_mixed_protocol"
        if normalized is not None:
            protocol_valid = True
            ordered = normalized[2] >= normalized[0] and normalized[3] >= normalized[1]
            in_bounds = all(0.0 <= value <= 1.0 for value in normalized)
            if ordered and in_bounds:
                iou = independent_iou(normalized, gt)
                hit = iou >= IOU_THRESHOLD
    return {
        "index": str(source_row["index"]),
        "prediction": prediction,
        "source_width": source_width,
        "source_height": source_height,
        "gt_normalized": gt,
        "parse_method": parse_method,
        "raw_coords": coords,
        "coordinate_space": coordinate_space,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "normalized_coords": normalized,
        "parsed": parsed,
        "protocol_valid": protocol_valid,
        "ordered": ordered,
        "in_bounds": in_bounds,
        "iou": iou,
        "hit": hit,
    }


def _optional_float_lists_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return len(left) == len(right) and all(_close(a, b) for a, b in zip(left, right))


def validate_detail_against_source(
    detail: dict[str, Any], source_row: dict[str, Any], source_width: int, source_height: int
) -> tuple[dict[str, Any], list[str]]:
    expected = independently_recompute_detail(source_row, source_width, source_height)
    errors: list[str] = []
    exact_fields = (
        "index",
        "prediction",
        "source_width",
        "source_height",
        "parse_method",
        "coordinate_space",
        "resized_width",
        "resized_height",
        "parsed",
        "protocol_valid",
        "ordered",
        "in_bounds",
        "hit",
    )
    for field in exact_fields:
        if detail.get(field) != expected[field]:
            errors.append(f"{field} mismatch")
    for field in ("gt_normalized", "raw_coords", "normalized_coords"):
        if not _optional_float_lists_equal(detail.get(field), expected[field]):
            errors.append(f"{field} mismatch")
    detail_iou = detail.get("iou")
    if not isinstance(detail_iou, (int, float)) or not _close(detail_iou, expected["iou"]):
        errors.append("iou mismatch")
    return expected, errors


def _close(left: float, right: float, *, atol: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)


def validate(summary_path: Path) -> dict[str, Any]:
    summary_path = summary_path.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    cells = summary.get("cells", [])
    cell_names = {str(cell.get("cell")) for cell in cells}
    if cell_names != EXPECTED_CELLS:
        errors.append(f"cell set mismatch: {sorted(cell_names)}")

    dimension_path = Path(str(summary.get("dimension_records", "")))
    if not dimension_path.is_file():
        errors.append(f"missing dimension records: {dimension_path}")
        dimensions = {}
    else:
        actual_sha = sha256_file(dimension_path)
        if actual_sha != summary.get("dimension_records_sha256"):
            errors.append("dimension-record SHA256 mismatch")
        dimensions = load_dimensions(dimension_path)
        if len(dimensions) != EXPECTED_SAMPLES:
            errors.append(f"dimension-record count is {len(dimensions)}")

    source_dimension_manifests = summary.get("source_dimension_manifests")
    if not isinstance(source_dimension_manifests, list) or not source_dimension_manifests:
        errors.append("missing source dimension manifests")
    else:
        for source in source_dimension_manifests:
            source_path = Path(str(source.get("path", "")))
            if not source_path.is_file():
                errors.append(f"missing source dimension manifest: {source_path}")
                continue
            if sha256_file(source_path) != source.get("sha256"):
                errors.append(f"source dimension manifest SHA256 mismatch: {source_path}")
                continue
            source_payload = json.loads(source_path.read_text(encoding="utf-8"))
            if source_payload.get("records_sha256") != summary.get("dimension_records_sha256"):
                errors.append(f"source dimension records SHA256 mismatch: {source_path}")
            if source_payload.get("selected_count") != EXPECTED_SAMPLES:
                errors.append(f"source dimension count mismatch: {source_path}")
            if source_payload.get("split_value") != "RefCOCOg_test":
                errors.append(f"source dimension split mismatch: {source_path}")

    total_rows = 0
    sentinel_rows: dict[str, dict[str, Any]] = {}
    validated_cells = []
    for cell in cells:
        name = str(cell.get("cell"))
        detail_path = Path(str(cell.get("details", "")))
        input_path = Path(str(cell.get("input", "")))
        if not detail_path.is_file() or not input_path.is_file():
            errors.append(f"{name}: missing input or detail artifact")
            continue
        if sha256_file(detail_path) != cell.get("details_sha256"):
            errors.append(f"{name}: detail SHA256 mismatch")
        if sha256_file(input_path) != cell.get("input_sha256"):
            errors.append(f"{name}: input SHA256 mismatch")
        source_manifest_path = Path(str(cell.get("source_prediction_manifest", "")))
        if not source_manifest_path.is_file():
            errors.append(f"{name}: missing source prediction manifest")
        elif sha256_file(source_manifest_path) != cell.get("source_prediction_manifest_sha256"):
            errors.append(f"{name}: source prediction manifest SHA256 mismatch")
        else:
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            expected_mode = "image_text" if cell.get("condition") == "IQ" else "image_text_image_text"
            source_task = source_manifest.get("task", {})
            source_checks = {
                "artifact_type": source_manifest.get("artifact_type") == "prediction",
                "status": source_manifest.get("status") == "complete",
                "expected_rows": source_manifest.get("expected_rows") == EXPECTED_SAMPLES,
                "prediction_sha256": source_manifest.get("prediction_sha256") == cell.get("input_sha256"),
                "model_key": source_task.get("model_key") == cell.get("model"),
                "mode": source_task.get("mode") == expected_mode,
                "dataset": source_task.get("dataset") == "RefCOCO",
            }
            failed = sorted(key for key, passed in source_checks.items() if not passed)
            if failed:
                errors.append(f"{name}: source prediction manifest failed checks {failed}")

        source_data = pd.read_excel(input_path)
        missing_columns = sorted(EXPECTED_COLUMNS - set(source_data.columns))
        if missing_columns:
            errors.append(f"{name}: source input missing columns {missing_columns}")
            continue
        if len(source_data) != EXPECTED_SAMPLES:
            errors.append(f"{name}: source input has {len(source_data)} rows")
            continue
        if source_data["index"].astype(str).duplicated().any():
            errors.append(f"{name}: source input has duplicate indices")
            continue
        if set(source_data["split"].astype(str)) != {"RefCOCOg_test"}:
            errors.append(f"{name}: source input has wrong split")
            continue
        source_rows = {
            str(source_row["index"]): source_row
            for source_row in source_data.to_dict("records")
        }

        indices: set[str] = set()
        parse_methods: Counter[str] = Counter()
        coordinate_spaces: Counter[str] = Counter()
        parsed = protocol_valid = ordered = in_bounds = native_recovered_hits = 0
        native_complete_hits = 0
        iou_sum = 0.0
        row_count = 0
        with gzip.open(detail_path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row_count += 1
                row = json.loads(line)
                index = str(row.get("index"))
                if index in indices:
                    errors.append(f"{name}: duplicate index {index}")
                    continue
                indices.add(index)
                if index not in dimensions or index not in source_rows:
                    errors.append(f"{name}: no dimension record for {index}")
                    continue
                expected_width, expected_height = dimensions[index]
                expected, row_errors = validate_detail_against_source(
                    row,
                    source_rows[index],
                    expected_width,
                    expected_height,
                )
                errors.extend(f"{name}:{index}: {message}" for message in row_errors)

                parse_method = expected["parse_method"]
                parse_methods[parse_method] += 1
                parsed_flag = expected["parsed"]
                protocol_valid_flag = expected["protocol_valid"]
                ordered_flag = expected["ordered"]
                in_bounds_flag = expected["in_bounds"]
                hit_flag = expected["hit"]
                coordinate_space = expected["coordinate_space"]
                iou = expected["iou"]
                if coordinate_space is not None:
                    coordinate_spaces[coordinate_space] += 1

                parsed += int(parsed_flag)
                protocol_valid += int(protocol_valid_flag)
                ordered += int(ordered_flag)
                in_bounds += int(in_bounds_flag)
                native_recovered_hits += int(hit_flag)
                if parse_method != "bbox_2d_open_array_recovery":
                    native_complete_hits += int(hit_flag)
                iou_sum += iou
                if name == "qwen25vl_7b__IQ" and index in {
                    "RefCOCOg_test_42959",
                    "RefCOCOg_test_42960",
                }:
                    sentinel_rows[index] = expected

        total_rows += row_count
        if row_count != EXPECTED_SAMPLES or len(indices) != EXPECTED_SAMPLES:
            errors.append(f"{name}: expected {EXPECTED_SAMPLES} unique rows, got {row_count}/{len(indices)}")
        expected_metrics = {
            "parsed": parsed,
            "protocol_valid": protocol_valid,
            "ordered": ordered,
            "in_bounds": in_bounds,
            "hits": native_complete_hits,
            "native_complete_hits": native_complete_hits,
            "native_recovered_hits": native_recovered_hits,
            "parse_success_pct": parsed / EXPECTED_SAMPLES * 100.0,
            "protocol_valid_pct": protocol_valid / EXPECTED_SAMPLES * 100.0,
            "ordered_box_pct": ordered / EXPECTED_SAMPLES * 100.0,
            "in_bounds_pct": in_bounds / EXPECTED_SAMPLES * 100.0,
            "precision_at_1": native_complete_hits / EXPECTED_SAMPLES * 100.0,
            "native_recovered_precision_at_1": native_recovered_hits / EXPECTED_SAMPLES * 100.0,
            "native_complete_precision_at_1": native_complete_hits / EXPECTED_SAMPLES * 100.0,
            "average_iou": iou_sum / EXPECTED_SAMPLES,
        }
        for key, expected in expected_metrics.items():
            actual = cell.get(key)
            if isinstance(expected, float):
                if actual is None or not _close(float(actual), expected):
                    errors.append(f"{name}: summary {key} mismatch")
            elif actual != expected:
                errors.append(f"{name}: summary {key} mismatch")
        if dict(sorted(parse_methods.items())) != cell.get("parse_method_counts"):
            errors.append(f"{name}: parse-method counts mismatch")
        if dict(sorted(coordinate_spaces.items())) != cell.get("coordinate_space_counts"):
            errors.append(f"{name}: coordinate-space counts mismatch")
        validated_cells.append(name)

    for index in ("RefCOCOg_test_42959", "RefCOCOg_test_42960"):
        row = sentinel_rows.get(index)
        if row is None:
            errors.append(f"missing sentinel {index}")
            continue
        if (row.get("resized_width"), row.get("resized_height")) != (1316, 784):
            errors.append(f"{index}: sentinel processed size mismatch")
        if float(row.get("iou", 0.0)) <= 0.95:
            errors.append(f"{index}: sentinel IoU did not exceed 0.95")

    return {
        "ok": not errors,
        "summary": str(summary_path),
        "cell_count": len(validated_cells),
        "total_detail_rows": total_rows,
        "dimension_record_count": len(dimensions),
        "sentinel_ious": {
            index: sentinel_rows[index]["iou"]
            for index in sorted(sentinel_rows)
        },
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = validate(args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
