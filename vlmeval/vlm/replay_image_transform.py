from __future__ import annotations

import base64
import hashlib
import io
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


BASELINE_IMAGE_TRANSFORM = "baseline"

SUPPORTED_IMAGE_TRANSFORMS = {
    BASELINE_IMAGE_TRANSFORM,
    "mask10_white",
    "mask10_black",
    "mask20_white",
    "mask20_black",
    "blank",
    "random_same_dataset",
    "rotate180",
    "shift_up_7_white",
    "shift_up_7_black",
    "shift_down_7_white",
    "shift_down_7_black",
    "shift_left_7_white",
    "shift_left_7_black",
    "shift_right_7_white",
    "shift_right_7_black",
    "shift_up_halfpatch_reflect",
    "shift_up_onepatch_reflect",
    "shift_down_halfpatch_reflect",
    "shift_down_onepatch_reflect",
    "shift_left_halfpatch_reflect",
    "shift_left_onepatch_reflect",
    "shift_right_halfpatch_reflect",
    "shift_right_onepatch_reflect",
    "shift_up_halfpatch_wrap",
    "shift_up_onepatch_wrap",
    "shift_down_halfpatch_wrap",
    "shift_down_onepatch_wrap",
    "shift_left_halfpatch_wrap",
    "shift_left_onepatch_wrap",
    "shift_right_halfpatch_wrap",
    "shift_right_onepatch_wrap",
    "shift_up_real_half_patch_wrap",
    "shift_down_real_half_patch_wrap",
    "shift_left_real_half_patch_wrap",
    "shift_right_real_half_patch_wrap",
    "zoom_1p5_uncropped",
}

MASK_GRID_ROWS = 10
MASK_GRID_COLS = 10
QWEN_PATCH_SIZE = 14
QWEN_SPATIAL_MERGE_SIZE = 2
QWEN_TOKEN_STRIDE = QWEN_PATCH_SIZE * QWEN_SPATIAL_MERGE_SIZE
QWEN_DEFAULT_MIN_PIXELS = 1280 * QWEN_TOKEN_STRIDE * QWEN_TOKEN_STRIDE
QWEN_DEFAULT_MAX_PIXELS = 16384 * QWEN_TOKEN_STRIDE * QWEN_TOKEN_STRIDE


def canonicalize_image_transform(name: str | None) -> str:
    raw = str(name or BASELINE_IMAGE_TRANSFORM).strip().lower()
    if raw not in SUPPORTED_IMAGE_TRANSFORMS:
        return BASELINE_IMAGE_TRANSFORM
    return raw


def image_transform_active(name: str | None) -> bool:
    return canonicalize_image_transform(name) != BASELINE_IMAGE_TRANSFORM


def _normalize_image_ref(path_or_url: str) -> str:
    raw = str(path_or_url or "").strip()
    if not raw:
        return raw
    if raw.startswith(("file://", "http://", "https://", "data:")):
        return raw
    return "file://" + raw


def _strip_file_scheme(path_or_url: str) -> str:
    raw = str(path_or_url or "").strip()
    if raw.startswith("file://"):
        return raw[len("file://") :]
    return raw


def _safe_token(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "sample"


def _fill_color(name: str) -> tuple[int, int, int]:
    if name == "black":
        return (0, 0, 0)
    return (255, 255, 255)


def _seed_from_parts(*parts: Any) -> int:
    payload = "||".join(str(part) for part in parts)
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _decode_data_url(data_url: str) -> Image.Image:
    header, payload = data_url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Only base64 data URLs are supported for replay image transforms.")
    raw = base64.b64decode(payload)
    with Image.open(io.BytesIO(raw)) as image:
        return image.convert("RGB")


def _load_image_from_ref(image_ref: str) -> Image.Image:
    raw = str(image_ref or "").strip()
    if raw.startswith("data:"):
        return _decode_data_url(raw)
    path = Path(_strip_file_scheme(raw))
    with Image.open(path) as image:
        return image.convert("RGB")


def _save_image(image: Image.Image, cache_dir: Path, cache_name: str) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / cache_name
    image.save(out_path)
    return _normalize_image_ref(str(out_path))


def _find_image_item_index(content: list[dict[str, Any]], image_position: int) -> tuple[int, dict[str, Any]]:
    current = 0
    for item_index, item in enumerate(content):
        if isinstance(item, dict) and item.get("type") == "image":
            current += 1
            if current == image_position:
                return item_index, item
    raise ValueError(f"Replay content only has {current} image item(s); cannot target image position {image_position}.")


def _shift_image(image: Image.Image, dx: int, dy: int, fill_color: tuple[int, int, int]) -> Image.Image:
    out = Image.new("RGB", image.size, color=fill_color)
    out.paste(image, (dx, dy))
    return out


def _smart_resize(height: int, width: int, *, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    if min(height, width) <= 0:
        raise ValueError("smart_resize requires positive image dimensions.")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / beta / factor) * factor)
        resized_width = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return resized_height, resized_width


def _estimate_qwen_llm_grid(
    image_size: tuple[int, int],
    *,
    min_pixels: int | None,
    max_pixels: int | None,
) -> dict[str, int]:
    width, height = image_size
    resized_height, resized_width = _smart_resize(
        height,
        width,
        factor=QWEN_TOKEN_STRIDE,
        min_pixels=int(min_pixels or QWEN_DEFAULT_MIN_PIXELS),
        max_pixels=int(max_pixels or QWEN_DEFAULT_MAX_PIXELS),
    )
    return {
        "resized_height": int(resized_height),
        "resized_width": int(resized_width),
        "llm_grid_h": max(1, int(resized_height // QWEN_TOKEN_STRIDE)),
        "llm_grid_w": max(1, int(resized_width // QWEN_TOKEN_STRIDE)),
    }


def _token_aware_shift_pixels(
    image_size: tuple[int, int],
    *,
    min_pixels: int | None,
    max_pixels: int | None,
    axis: str,
    patch_fraction: float,
) -> dict[str, Any]:
    width, height = image_size
    grid_meta = _estimate_qwen_llm_grid(image_size, min_pixels=min_pixels, max_pixels=max_pixels)
    if axis == "x":
        patch_extent = float(width) / float(grid_meta["llm_grid_w"])
        shift_pixels = max(1, int(round(patch_extent * patch_fraction)))
    elif axis == "y":
        patch_extent = float(height) / float(grid_meta["llm_grid_h"])
        shift_pixels = max(1, int(round(patch_extent * patch_fraction)))
    else:
        raise ValueError(f"Unsupported axis for token-aware shift: {axis}")
    return {
        "shift_pixels": int(shift_pixels),
        "patch_extent": float(patch_extent),
        **grid_meta,
    }


def _shift_image_reflect(image: Image.Image, *, dx: int, dy: int) -> Image.Image:
    if dx == 0 and dy == 0:
        return image.copy()
    width, height = image.size
    left = max(dx, 0)
    right = max(-dx, 0)
    top = max(dy, 0)
    bottom = max(-dy, 0)
    source = np.asarray(image.convert("RGB"))
    padded = np.pad(source, ((top, bottom), (left, right), (0, 0)), mode="reflect")
    start_x = left - dx
    start_y = top - dy
    cropped = padded[start_y : start_y + height, start_x : start_x + width]
    return Image.fromarray(cropped.astype(np.uint8), mode="RGB")


def _shift_image_wrap(image: Image.Image, *, dx: int, dy: int) -> Image.Image:
    if dx == 0 and dy == 0:
        return image.copy()
    source = np.asarray(image.convert("RGB"))
    shifted = np.roll(source, shift=dy, axis=0)
    shifted = np.roll(shifted, shift=dx, axis=1)
    return Image.fromarray(shifted.astype(np.uint8), mode="RGB")


def _mask_image(
    image: Image.Image,
    *,
    ratio: float,
    fill_color: tuple[int, int, int],
    seed: int,
) -> tuple[Image.Image, list[dict[str, int]]]:
    width, height = image.size
    total_cells = MASK_GRID_ROWS * MASK_GRID_COLS
    masked_cells = max(1, round(total_cells * ratio))
    rng = random.Random(seed)
    selected = sorted(rng.sample(range(total_cells), masked_cells))
    out = image.copy()
    draw = ImageDraw.Draw(out)
    cell_boxes: list[dict[str, int]] = []
    for cell_id in selected:
        row = cell_id // MASK_GRID_COLS
        col = cell_id % MASK_GRID_COLS
        x0 = round((col / MASK_GRID_COLS) * width)
        x1 = round(((col + 1) / MASK_GRID_COLS) * width)
        y0 = round((row / MASK_GRID_ROWS) * height)
        y1 = round(((row + 1) / MASK_GRID_ROWS) * height)
        draw.rectangle((x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)), fill=fill_color)
        cell_boxes.append({"cell_id": cell_id, "row": row, "col": col, "x0": x0, "x1": x1, "y0": y0, "y1": y1})
    return out, cell_boxes


def _resize_to_match(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    return image.resize(target_size, resample=Image.Resampling.BICUBIC)


def apply_image_transform_to_content(
    content: list[dict[str, Any]],
    *,
    transform_name: str,
    sample_meta: dict[str, Any] | None,
    cache_dir: str | Path,
    dataset_name: str,
    image_position: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transform = canonicalize_image_transform(transform_name)
    out = [dict(item) if isinstance(item, dict) else item for item in content]
    if transform == BASELINE_IMAGE_TRANSFORM:
        return out, {"transform": transform, "applied": False}

    item_index, image_item = _find_image_item_index(out, image_position=image_position)
    original_ref = str(image_item.get("image", "")).strip()
    if not original_ref:
        raise ValueError(f"Image position {image_position} does not have a usable `image` ref.")

    sample_meta = dict(sample_meta or {})
    sample_index = sample_meta.get("sample_index", "unknown")
    sample_key = _safe_token(f"{dataset_name}_{sample_index}_{transform}")
    cache_dir = Path(cache_dir)

    record: dict[str, Any] = {
        "transform": transform,
        "applied": True,
        "dataset": dataset_name,
        "sample_index": str(sample_index),
        "target_image_position": image_position,
        "content_item_index": item_index,
        "original_image_ref": original_ref,
    }

    source_image = _load_image_from_ref(original_ref)
    record["original_image_size"] = list(source_image.size)
    output_image = source_image
    output_ref = original_ref

    if transform == "blank":
        output_image = Image.new("RGB", source_image.size, color=_fill_color("white"))
    elif transform == "random_same_dataset":
        donor_ref = str(sample_meta.get("random_same_dataset_image_ref", "")).strip()
        donor_index = sample_meta.get("random_same_dataset_donor_index")
        if not donor_ref:
            raise ValueError("random_same_dataset transform requires `random_same_dataset_image_ref` in replay_meta.")
        donor_image = _load_image_from_ref(donor_ref)
        record["donor_image_ref"] = donor_ref
        record["donor_image_size"] = list(donor_image.size)
        record["donor_index"] = None if donor_index is None else str(donor_index)
        output_image = _resize_to_match(donor_image, source_image.size)
    elif transform == "rotate180":
        output_image = source_image.transpose(Image.Transpose.ROTATE_180)
    elif transform == "zoom_1p5_uncropped":
        width, height = source_image.size
        output_image = source_image.resize(
            (max(1, round(width * 1.5)), max(1, round(height * 1.5))),
            resample=Image.Resampling.BICUBIC,
        )
    elif transform.startswith("shift_") and transform.endswith("_real_half_patch_wrap"):
        parts = transform.split("_")
        if len(parts) != 6:
            raise ValueError(f"Unsupported raw real-half-patch shift transform: {transform}")
        _, direction, _, _, _, pad_mode = parts
        if pad_mode != "wrap":
            raise ValueError(f"Unsupported raw real-half-patch padding mode: {pad_mode}")
        axis = "x" if direction in {"left", "right"} else "y"
        shift_info = _token_aware_shift_pixels(
            source_image.size,
            min_pixels=image_item.get("min_pixels"),
            max_pixels=image_item.get("max_pixels"),
            axis=axis,
            patch_fraction=0.5,
        )
        shift_px = 7
        dx = 0
        dy = 0
        if direction == "up":
            dy = -shift_px
        elif direction == "down":
            dy = shift_px
        elif direction == "left":
            dx = -shift_px
        elif direction == "right":
            dx = shift_px
        else:
            raise ValueError(f"Unsupported raw real-half-patch shift direction: {direction}")
        output_image = _shift_image_wrap(source_image, dx=dx, dy=dy)
        record["shift"] = {
            "dx": dx,
            "dy": dy,
            "pad_mode": pad_mode,
            "raw_pixel_shift": shift_px,
            "pixel_shift_kind": "raw_real_half_patch",
            "reference_vit_patch_size": QWEN_PATCH_SIZE,
            "reference_half_vit_patch_size": QWEN_PATCH_SIZE // 2,
            "estimated_patch_extent": shift_info["patch_extent"],
            "estimated_resized_height": shift_info["resized_height"],
            "estimated_resized_width": shift_info["resized_width"],
            "estimated_llm_grid_h": shift_info["llm_grid_h"],
            "estimated_llm_grid_w": shift_info["llm_grid_w"],
            "qwen_patch_size": QWEN_PATCH_SIZE,
            "qwen_spatial_merge_size": QWEN_SPATIAL_MERGE_SIZE,
            "qwen_token_stride": QWEN_TOKEN_STRIDE,
        }
    elif (transform.endswith("_reflect") or transform.endswith("_wrap")) and transform.startswith("shift_"):
        parts = transform.split("_")
        if len(parts) != 4:
            raise ValueError(f"Unsupported token-aware shift transform: {transform}")
        _, direction, magnitude_name, pad_mode = parts
        if pad_mode not in {"reflect", "wrap"}:
            raise ValueError(f"Unsupported token-aware shift padding mode: {pad_mode}")
        patch_fraction = {"halfpatch": 0.5, "onepatch": 1.0}.get(magnitude_name)
        if patch_fraction is None:
            raise ValueError(f"Unsupported token-aware shift magnitude: {magnitude_name}")
        axis = "x" if direction in {"left", "right"} else "y"
        shift_info = _token_aware_shift_pixels(
            source_image.size,
            min_pixels=image_item.get("min_pixels"),
            max_pixels=image_item.get("max_pixels"),
            axis=axis,
            patch_fraction=patch_fraction,
        )
        shift_px = int(shift_info["shift_pixels"])
        dx = 0
        dy = 0
        if direction == "up":
            dy = -shift_px
        elif direction == "down":
            dy = shift_px
        elif direction == "left":
            dx = -shift_px
        elif direction == "right":
            dx = shift_px
        else:
            raise ValueError(f"Unsupported token-aware shift direction: {direction}")
        if pad_mode == "reflect":
            output_image = _shift_image_reflect(source_image, dx=dx, dy=dy)
        else:
            output_image = _shift_image_wrap(source_image, dx=dx, dy=dy)
        record["shift"] = {
            "dx": dx,
            "dy": dy,
            "pad_mode": pad_mode,
            "patch_fraction": patch_fraction,
            "estimated_patch_extent": shift_info["patch_extent"],
            "estimated_resized_height": shift_info["resized_height"],
            "estimated_resized_width": shift_info["resized_width"],
            "estimated_llm_grid_h": shift_info["llm_grid_h"],
            "estimated_llm_grid_w": shift_info["llm_grid_w"],
            "qwen_patch_size": QWEN_PATCH_SIZE,
            "qwen_spatial_merge_size": QWEN_SPATIAL_MERGE_SIZE,
            "qwen_token_stride": QWEN_TOKEN_STRIDE,
        }
    elif transform.startswith("shift_"):
        direction, fill_name = transform.rsplit("_", 1)
        fill_color = _fill_color(fill_name)
        shift_px = 7
        dx = 0
        dy = 0
        if direction == "shift_up_7":
            dy = -shift_px
        elif direction == "shift_down_7":
            dy = shift_px
        elif direction == "shift_left_7":
            dx = -shift_px
        elif direction == "shift_right_7":
            dx = shift_px
        else:
            raise ValueError(f"Unsupported shift transform: {transform}")
        output_image = _shift_image(source_image, dx=dx, dy=dy, fill_color=fill_color)
        record["shift"] = {"dx": dx, "dy": dy, "fill": fill_name}
    elif transform.startswith("mask"):
        mask_prefix, fill_name = transform.split("_", 1)
        ratio = 0.10 if mask_prefix == "mask10" else 0.20
        fill_color = _fill_color(fill_name)
        seed = _seed_from_parts(dataset_name, sample_index, transform)
        output_image, masked_cells = _mask_image(
            source_image,
            ratio=ratio,
            fill_color=fill_color,
            seed=seed,
        )
        record["mask"] = {
            "ratio": ratio,
            "grid_rows": MASK_GRID_ROWS,
            "grid_cols": MASK_GRID_COLS,
            "seed": seed,
            "masked_cell_count": len(masked_cells),
            "masked_cells": masked_cells,
            "fill": fill_name,
        }
    else:
        raise ValueError(f"Unsupported image transform: {transform}")

    if transform != "baseline":
        output_ref = _save_image(output_image, cache_dir, f"{sample_key}.png")

    out[item_index]["image"] = output_ref
    record["transformed_image_ref"] = output_ref
    record["transformed_image_size"] = list(output_image.size)
    return out, record
