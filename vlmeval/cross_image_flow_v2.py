from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class ImageGridMeta:
    image_index: int
    temporal: int
    grid_h: int
    grid_w: int
    spatial_merge_size: int

    @property
    def llm_grid_h(self) -> int:
        return self.grid_h // self.spatial_merge_size

    @property
    def llm_grid_w(self) -> int:
        return self.grid_w // self.spatial_merge_size

    @property
    def token_count(self) -> int:
        return self.temporal * self.llm_grid_h * self.llm_grid_w

    def to_dict(self) -> dict[str, int]:
        return {
            "image_index": int(self.image_index),
            "temporal": int(self.temporal),
            "grid_h": int(self.grid_h),
            "grid_w": int(self.grid_w),
            "spatial_merge_size": int(self.spatial_merge_size),
            "llm_grid_h": int(self.llm_grid_h),
            "llm_grid_w": int(self.llm_grid_w),
            "token_count": int(self.token_count),
        }


def load_controlled_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Controlled manifest must be a list: {manifest_path}")
    for item in data:
        if "id" not in item or "image" not in item or "questions" not in item:
            raise ValueError(f"Malformed manifest item: {item}")
    return data


def resolve_manifest_image_path(manifest_path: str | Path, image_ref: str) -> Path:
    base = Path(manifest_path).resolve().parent
    image_path = (base / image_ref).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Manifest image not found: {image_path}")
    return image_path


def flatten_controlled_manifest(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for item in manifest:
        for question in item["questions"]:
            target_box = question["target_box_xyxy"]
            if len(target_box) != 4:
                raise ValueError(f"target_box_xyxy must have 4 elements: {question}")
            cases.append(
                {
                    "base_id": item["id"],
                    "question_id": question["id"],
                    "case_id": f"{item['id']}__{question['id']}",
                    "image": item["image"],
                    "source": item.get("source", {}),
                    "kind": question.get("kind", ""),
                    "question": question["question"],
                    "answer": question.get("answer", ""),
                    "target_box_xyxy": [int(x) for x in target_box],
                }
            )
    return cases


def extract_image_grid_meta(
    image_grid_thw: torch.Tensor | np.ndarray | list[list[int]],
    *,
    spatial_merge_size: int,
) -> list[ImageGridMeta]:
    if image_grid_thw is None:
        raise ValueError("image_grid_thw is required for image-grid reconstruction.")
    if torch.is_tensor(image_grid_thw):
        values = image_grid_thw.detach().cpu().tolist()
    else:
        values = image_grid_thw
    metas: list[ImageGridMeta] = []
    for image_index, (temporal, grid_h, grid_w) in enumerate(values):
        metas.append(
            ImageGridMeta(
                image_index=int(image_index),
                temporal=int(temporal),
                grid_h=int(grid_h),
                grid_w=int(grid_w),
                spatial_merge_size=int(spatial_merge_size),
            )
        )
    return metas


def token_rows_and_cols(meta: ImageGridMeta) -> tuple[np.ndarray, np.ndarray]:
    rows = np.repeat(np.arange(meta.llm_grid_h, dtype=np.int32), meta.llm_grid_w)
    cols = np.tile(np.arange(meta.llm_grid_w, dtype=np.int32), meta.llm_grid_h)
    if meta.temporal != 1:
        rows = np.tile(rows, meta.temporal)
        cols = np.tile(cols, meta.temporal)
    return rows, cols


def image_token_table(meta: ImageGridMeta) -> list[dict[str, int]]:
    rows, cols = token_rows_and_cols(meta)
    table = []
    for token_index, (row, col) in enumerate(zip(rows.tolist(), cols.tolist())):
        table.append(
            {
                "token_index": int(token_index),
                "row": int(row),
                "col": int(col),
            }
        )
    return table


def bbox_to_token_indices(
    *,
    image_size: tuple[int, int],
    grid_meta: ImageGridMeta,
    bbox_xyxy: list[int],
) -> list[int]:
    width, height = image_size
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    if width <= 0 or height <= 0:
        return []
    token_w = width / max(grid_meta.llm_grid_w, 1)
    token_h = height / max(grid_meta.llm_grid_h, 1)
    indices: list[int] = []
    for token_index, entry in enumerate(image_token_table(grid_meta)):
        cx = (entry["col"] + 0.5) * token_w
        cy = (entry["row"] + 0.5) * token_h
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            indices.append(int(token_index))
    return indices


def normalize_rows(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = matrix.sum(axis=-1, keepdims=True)
    denom = np.maximum(denom, eps)
    return matrix / denom


def safe_float16(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float16)
