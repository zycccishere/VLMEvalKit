#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _pick_first(rows: list[dict[str, Any]], phase: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("phase") == phase:
            return row
    return None


def _load_rgb(path_or_ref: str) -> np.ndarray:
    raw = str(path_or_ref or "").strip()
    if raw.startswith("file://"):
        raw = raw[len("file://") :]
    with Image.open(raw) as image:
        return np.asarray(image.convert("RGB"))


def _wrap_verified(original: np.ndarray, transformed: np.ndarray, dx: int, dy: int) -> bool:
    if dx > 0:
        return np.array_equal(transformed[:, :dx, :], original[:, -dx:, :])
    if dx < 0:
        k = abs(dx)
        return np.array_equal(transformed[:, -k:, :], original[:, :k, :])
    if dy > 0:
        return np.array_equal(transformed[:dy, :, :], original[-dy:, :, :])
    if dy < 0:
        k = abs(dy)
        return np.array_equal(transformed[-k:, :, :], original[:k, :, :])
    return True


def _expected_direction_ok(transform: str, dx: int, dy: int) -> bool:
    if "shift_left_" in transform:
        return dx < 0 and dy == 0
    if "shift_right_" in transform:
        return dx > 0 and dy == 0
    if "shift_up_" in transform:
        return dy < 0 and dx == 0
    if "shift_down_" in transform:
        return dy > 0 and dx == 0
    return False


def _summarize_trace(trace_file: Path) -> list[dict[str, Any]]:
    transform = trace_file.parents[2].name
    rows = _load_jsonl(trace_file)
    image_transform = _pick_first(rows, "image_transform") or {}
    record = image_transform.get("record", {}) if isinstance(image_transform.get("record"), dict) else {}
    shift = record.get("shift", {}) if isinstance(record.get("shift"), dict) else {}
    if not record:
        return []
    original_ref = str(record.get("original_image_ref", "")).strip()
    transformed_ref = str(record.get("transformed_image_ref", "")).strip()
    original = _load_rgb(original_ref)
    transformed = _load_rgb(transformed_ref)
    dx = int(shift.get("dx", 0) or 0)
    dy = int(shift.get("dy", 0) or 0)
    estimated_patch_extent = shift.get("estimated_patch_extent")
    halfpatch_ratio = None
    if estimated_patch_extent not in {None, 0, 0.0}:
        halfpatch_ratio = (abs(dx) + abs(dy)) / float(estimated_patch_extent)
    return [
        {
            "trace_file": str(trace_file),
            "dataset": image_transform.get("dataset"),
            "transform": transform,
            "sample_index": record.get("sample_index"),
            "original_image_size": record.get("original_image_size"),
            "transformed_image_size": record.get("transformed_image_size"),
            "dx": dx,
            "dy": dy,
            "pad_mode": shift.get("pad_mode"),
            "pixel_shift_kind": shift.get("pixel_shift_kind", "token_aware"),
            "raw_pixel_shift": shift.get("raw_pixel_shift"),
            "estimated_patch_extent": estimated_patch_extent,
            "halfpatch_ratio": halfpatch_ratio,
            "direction_ok": _expected_direction_ok(transform, dx, dy),
            "wrap_ok": shift.get("pad_mode") == "wrap",
            "border_wrap_verified": _wrap_verified(original, transformed, dx, dy),
            "reference_vit_patch_size": shift.get("reference_vit_patch_size"),
            "reference_half_vit_patch_size": shift.get("reference_half_vit_patch_size"),
            "qwen_token_stride": shift.get("qwen_token_stride"),
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Qwen image2 transform smoke traces.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    root = Path(args.results_root)
    trace_files = sorted(root.glob("default/image_text_image/*/qwen25vl_32b/_trace/Qwen2VLChatReplay.jsonl"))
    rows: list[dict[str, Any]] = []
    for trace_file in trace_files:
        rows.extend(_summarize_trace(trace_file))
    if args.pretty:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(rows, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
