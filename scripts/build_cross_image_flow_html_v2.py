#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

MODE_LABELS = {
    "image_text_image": "IQI",
    "image_image_text": "IIQ",
}
MODE_IQI = "image_text_image"
MODE_IIQ = "image_image_text"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an IQI vs IIQ cross-image flow HTML viewer.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--float-decimals",
        type=int,
        default=5,
        help="Round exported float matrices to this many decimals to keep bundle size reasonable.",
    )
    return parser


def resolve_image_path(summary_path: Path, image_ref: str) -> Path:
    source_image = Path(image_ref)
    candidates = [
        source_image,
        summary_path.parent / source_image.name,
        summary_path.parent.parent / "cross-image-flow" / "images" / source_image.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in summary_path.parent.parent.rglob(source_image.name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Unable to resolve viewer image: {image_ref}")


def round_matrix(values: np.ndarray, decimals: int) -> list[Any]:
    return np.round(np.asarray(values, dtype=np.float32), decimals=decimals).tolist()


def case_bundle_filename(case_id: str) -> str:
    return f"{case_id}.js"


def mode_label(mode_name: str) -> str:
    return MODE_LABELS.get(mode_name, mode_name.replace("_", " ").upper())


def preferred_mode_order(mode_names: list[str]) -> list[str]:
    preferred = ["image_text_image", "image_image_text"]
    ordered = [mode for mode in preferred if mode in mode_names]
    ordered.extend([mode for mode in mode_names if mode not in ordered])
    return ordered


def mode_payload(mode_name: str, payload: dict[str, Any], decimals: int) -> dict[str, Any]:
    return {
        "mode": mode_name,
        "mode_label": mode_label(mode_name),
        "prompt_text": payload.get("prompt_text", ""),
        "input_token_count": int(payload.get("input_token_count", 0)),
        "image_spans": payload.get("image_spans", []),
        "image_grid_metas": payload.get("image_grid_metas", []),
        "image_size": payload.get("image_size", {}),
        "target_box": payload.get("target_box", {}),
        "raw_grid_shape": payload.get("raw_grid_shape", {}),
        "viewer_grid_shape": payload.get("viewer_grid_shape", {}),
        "target_query_patch_indices": payload.get("target_query_patch_indices", []),
        "target_key_patch_indices": payload.get("target_key_patch_indices", []),
        "image1_patch_boxes": payload.get("image1_patch_boxes", []),
        "image2_patch_boxes": payload.get("image2_patch_boxes", []),
        "selected_layers": payload.get("selected_layers", []),
        "normalized_q_to_k": round_matrix(np.asarray(payload["normalized_q_to_k"], dtype=np.float32), decimals),
        "summary_metrics": payload.get("summary_metrics", {}),
        "generated": payload.get("generated", {}),
    }


def patch_boxes_from_token_table(
    token_table: list[dict[str, Any]],
    image_size: list[int] | tuple[int, int],
    grid_info: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    width, height = [int(v) for v in image_size]
    if grid_info is not None:
        grid_h = int(grid_info.get("llm_grid_h") or 0)
        grid_w = int(grid_info.get("llm_grid_w") or 0)
    else:
        grid_h = 0
        grid_w = 0
    if not grid_h or not grid_w:
        for entry in token_table:
            grid_h = max(grid_h, int(entry.get("row", 0)) + 1)
            grid_w = max(grid_w, int(entry.get("col", 0)) + 1)
    cell_w = width / max(grid_w, 1)
    cell_h = height / max(grid_h, 1)
    boxes: list[dict[str, Any]] = []
    for entry in token_table:
        row = int(entry["row"])
        col = int(entry["col"])
        boxes.append(
            {
                "token_index": int(entry.get("token_index", len(boxes))),
                "row": row,
                "col": col,
                "x": float(col * cell_w),
                "y": float(row * cell_h),
                "w": float(cell_w),
                "h": float(cell_h),
            }
        )
    return boxes


def summary_metrics_from_layer_summaries(layer_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not layer_summaries:
        return {}

    def mean_for(key: str) -> float:
        values = [float(item[key]) for item in layer_summaries if item.get(key) is not None]
        if not values:
            return float("nan")
        return float(np.mean(np.asarray(values, dtype=np.float32)))

    return {
        "mean_cross_image_mass": mean_for("mean_image1_mass_raw"),
        "mean_target_box_mass_full_query": mean_for("target_box_mass_norm_all_queries"),
        "mean_target_box_mass_target_query": mean_for("target_box_mass_norm_target_queries"),
        "mean_diag_mass_full_query": mean_for("diag_mass_norm_all_queries"),
        "mean_diag_mass_target_query": mean_for("diag_mass_norm_other_queries"),
    }


def grouped_case_payload(
    *,
    summary_path: Path,
    case_id: str,
    records: list[dict[str, Any]],
    decimals: int,
) -> tuple[dict[str, Any], Path]:
    exemplar = records[0]
    image_path = resolve_image_path(summary_path, exemplar.get("image") or exemplar.get("image_path", ""))
    mode_map: dict[str, Any] = {}
    for record in records:
        npz_path = Path(record["npz_path"])
        if not npz_path.is_absolute():
            npz_path = (summary_path.parent / npz_path).resolve()
        blob = np.load(npz_path, allow_pickle=True)
        normalized_q_to_k = np.asarray(blob["layer_mean_norm"], dtype=np.float32)
        image1_grid = record.get("image1_grid") or {}
        image2_grid = record.get("image2_grid") or {}
        image_size = record.get("image_size") or exemplar.get("image_size") or []
        mode_map[record["mode"]] = {
            "prompt_text": record.get("prompt_text", ""),
            "input_token_count": 0,
            "image_spans": [],
            "image_grid_metas": [image1_grid, image2_grid],
            "image_size": image_size,
            "target_box": record.get("target_box_xyxy", []),
            "raw_grid_shape": {
                "image1": {
                    "h": int(image1_grid.get("llm_grid_h", 0)),
                    "w": int(image1_grid.get("llm_grid_w", 0)),
                },
                "image2": {
                    "h": int(image2_grid.get("llm_grid_h", 0)),
                    "w": int(image2_grid.get("llm_grid_w", 0)),
                },
            },
            "viewer_grid_shape": {
                "image1": {
                    "h": int(image1_grid.get("llm_grid_h", 0)),
                    "w": int(image1_grid.get("llm_grid_w", 0)),
                },
                "image2": {
                    "h": int(image2_grid.get("llm_grid_h", 0)),
                    "w": int(image2_grid.get("llm_grid_w", 0)),
                },
            },
            "target_query_patch_indices": record.get("target_query_token_indices", []),
            "target_key_patch_indices": record.get("target_key_token_indices", []),
            "image1_patch_boxes": patch_boxes_from_token_table(
                record.get("image1_token_table", []),
                image_size,
                image1_grid,
            ),
            "image2_patch_boxes": patch_boxes_from_token_table(
                record.get("image2_token_table", []),
                image_size,
                image2_grid,
            ),
            "selected_layers": record.get("selected_layers", []),
            "normalized_q_to_k": normalized_q_to_k,
            "summary_metrics": summary_metrics_from_layer_summaries(record.get("layer_summaries", [])),
            "generated": {
                "text": record.get("generated_text", ""),
                "ids": record.get("generated_ids", []),
            },
        }
    mode_names = preferred_mode_order(list(mode_map.keys()))
    payload = {
        "case_id": case_id,
        "base_id": exemplar.get("base_id", case_id),
        "question_id": exemplar.get("question_id", case_id),
        "question": exemplar.get("question", ""),
        "target_label": exemplar.get("answer", ""),
        "group_id": exemplar.get("base_id", case_id),
        "group_type": exemplar.get("kind", ""),
        "source": exemplar.get("source", {}),
        "image_relpath": f"images/{image_path.name}",
        "image_size": exemplar.get("image_size", []),
        "target_box_xyxy": exemplar.get("target_box_xyxy", []),
        "mode_order": mode_names,
        "modes": {
            mode_name: mode_payload(mode_name, mode_map[mode_name], decimals)
            for mode_name in mode_names
        },
    }
    return payload, image_path


def build_html(index_payload: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IQI vs IIQ Cross-Image Flow Viewer v2</title>
  <style>
    :root {{
      --bg: #f6efe3;
      --card: #fffaf2;
      --line: #d8cebf;
      --ink: #1e1b18;
      --muted: #72695f;
      --accent: #0f7c66;
      --accent-2: #b24e33;
      --accent-3: #4d72b0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top right, rgba(15,124,102,0.10), transparent 24%),
        radial-gradient(circle at bottom left, rgba(178,78,51,0.10), transparent 22%),
        var(--bg);
    }}
    .page {{ max-width: 1700px; margin: 0 auto; padding: 22px; }}
    .top {{
      display: grid;
      grid-template-columns: 1.25fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 8px 28px rgba(43,31,24,0.06);
    }}
    .question {{ padding: 18px 22px; }}
    .eyebrow {{
      color: var(--accent);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      margin-bottom: 10px;
    }}
    .question-main {{
      font-size: 28px;
      line-height: 1.18;
      font-weight: 700;
      margin-bottom: 12px;
    }}
    .question-meta {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      font-size: 13px;
      color: var(--muted);
    }}
    .question-meta strong {{
      display: block;
      color: var(--ink);
      font-size: 16px;
      margin-top: 2px;
    }}
    .controls {{
      padding: 16px 18px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      align-content: start;
    }}
    .control label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    select {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #fff;
      font: inherit;
    }}
    .mode-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }}
    .panel {{
      padding: 12px;
      display: grid;
      gap: 10px;
    }}
    .panel-title {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 18px;
      font-weight: 700;
      padding: 4px 6px 0;
    }}
    .panel-title span {{
      font-size: 12px;
      font-weight: 500;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .stack {{
      display: grid;
      gap: 8px;
    }}
    .canvas-wrap {{
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: #fff;
      display: grid;
      place-items: center;
      min-height: 280px;
    }}
    canvas {{
      display: block;
      max-width: 100%;
      height: auto;
      cursor: crosshair;
    }}
    .surface-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 16px;
    }}
    .surface {{
      padding: 12px;
      display: grid;
      gap: 10px;
    }}
    .surface canvas {{
      width: 100%;
      height: auto;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(8, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .metric {{
      padding: 14px 16px;
      border-radius: 14px;
      background: var(--card);
      border: 1px solid var(--line);
    }}
    .metric .k {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 4px;
    }}
    .metric .v {{
      font-size: 20px;
      font-weight: 700;
    }}
    .notes {{
      padding: 16px 18px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      font-size: 14px;
      line-height: 1.45;
      color: var(--muted);
    }}
    .notes strong {{
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent);
      margin-bottom: 6px;
    }}
    .delta-note {{
      font-size: 13px;
      color: var(--muted);
      line-height: 1.45;
    }}
    @media (max-width: 1180px) {{
      .top,
      .mode-grid,
      .surface-grid,
      .metrics,
      .question-meta,
      .notes,
      .controls {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="top">
      <div class="card question">
        <div class="eyebrow">Controlled Pack Viewer</div>
        <div id="questionText" class="question-main"></div>
        <div class="question-meta">
          <div>Target Answer<strong id="answerText">-</strong></div>
          <div>Dataset<strong id="datasetText">-</strong></div>
          <div>Base Image<strong id="baseText">-</strong></div>
          <div>Group Type<strong id="kindText">-</strong></div>
          <div>Mode Pair<strong id="modePairText">-</strong></div>
        </div>
      </div>
      <div class="card controls">
        <div class="control">
          <label for="caseSelect">Case</label>
          <select id="caseSelect"></select>
        </div>
        <div class="control">
          <label for="scopeSelect">Query Scope</label>
          <select id="scopeSelect">
            <option value="patch">Selected patch</option>
            <option value="target_mean">Target-box mean</option>
            <option value="full_mean">Full-image mean</option>
          </select>
        </div>
        <div class="control">
          <label for="massSelect">Contour Mass</label>
          <select id="massSelect">
            <option value="0.5">50%</option>
            <option value="0.6">60%</option>
            <option value="0.7">70%</option>
          </select>
        </div>
        <div class="control">
          <label for="alphaSelect">Heat Alpha</label>
          <select id="alphaSelect">
            <option value="0.55">0.55</option>
            <option value="0.7">0.70</option>
            <option value="0.85">0.85</option>
          </select>
        </div>
        <div class="control">
          <label for="targetSelect">Target Overlay</label>
          <select id="targetSelect">
            <option value="on">On</option>
            <option value="off">Off</option>
          </select>
        </div>
      </div>
    </div>

    <div class="mode-grid">
      <div class="card panel">
        <div class="panel-title"><span>IQI / image_text_image</span><span>query + map</span></div>
        <div class="stack">
          <div class="canvas-wrap"><canvas id="iqiQuery"></canvas></div>
          <div class="canvas-wrap"><canvas id="iqiMap"></canvas></div>
        </div>
      </div>
      <div class="card panel">
        <div class="panel-title"><span>IIQ / image_image_text</span><span>query + map</span></div>
        <div class="stack">
          <div class="canvas-wrap"><canvas id="iiqQuery"></canvas></div>
          <div class="canvas-wrap"><canvas id="iiqMap"></canvas></div>
        </div>
      </div>
      <div class="card panel">
        <div class="panel-title"><span>Delta</span><span>IQI - IIQ on image1</span></div>
        <div class="stack">
          <div class="canvas-wrap"><canvas id="deltaMap"></canvas></div>
          <div class="delta-note">
            Positive values mean IQI puts more mass on the same key patch than IIQ. Negative values mean IIQ is stronger.
          </div>
        </div>
      </div>
    </div>

    <div class="surface-grid">
      <div class="card surface">
        <div class="panel-title"><span>IQI surface</span><span>layer-mean normalized map</span></div>
        <div class="canvas-wrap"><canvas id="iqiSurface"></canvas></div>
      </div>
      <div class="card surface">
        <div class="panel-title"><span>IIQ surface</span><span>layer-mean normalized map</span></div>
        <div class="canvas-wrap"><canvas id="iiqSurface"></canvas></div>
      </div>
      <div class="card surface">
        <div class="panel-title"><span>Delta surface</span><span>IQI - IIQ</span></div>
        <div class="canvas-wrap"><canvas id="deltaSurface"></canvas></div>
      </div>
    </div>

    <div class="metrics">
      <div class="metric"><div class="k">Query Scope</div><div id="metricScope" class="v">-</div></div>
      <div class="metric"><div class="k">IQI Target Mass</div><div id="metricIQITarget" class="v">-</div></div>
      <div class="metric"><div class="k">IIQ Target Mass</div><div id="metricIIQTarget" class="v">-</div></div>
      <div class="metric"><div class="k">Delta Target Mass</div><div id="metricDeltaTarget" class="v">-</div></div>
      <div class="metric"><div class="k">IQI Diag Mass</div><div id="metricIQIDiag" class="v">-</div></div>
      <div class="metric"><div class="k">IIQ Diag Mass</div><div id="metricIIQDiag" class="v">-</div></div>
      <div class="metric"><div class="k">Delta Diag Mass</div><div id="metricDeltaDiag" class="v">-</div></div>
      <div class="metric"><div class="k">JSD IQI/IIQ</div><div id="metricJSD" class="v">-</div></div>
    </div>

    <div class="card notes">
      <div>
        <strong>How To Read</strong>
        The left and middle columns show the same image with the same query selection under the two settings. The right column shows the direct difference map on image1, with positive values favoring IQI.
      </div>
      <div>
        <strong>Interpretation Hint</strong>
        The summary metrics are computed on the same scope as the map view. The surface plots are coarse pooled versions of the layer-mean normalized q-to-k matrix, so they show shape rather than per-patch noise.
      </div>
    </div>
  </div>

  <script>
    const INDEX = {json.dumps(index_payload, ensure_ascii=False)};
    const caseSelect = document.getElementById("caseSelect");
    const scopeSelect = document.getElementById("scopeSelect");
    const massSelect = document.getElementById("massSelect");
    const alphaSelect = document.getElementById("alphaSelect");
    const targetSelect = document.getElementById("targetSelect");
    const canvases = {{
      iqiQuery: document.getElementById("iqiQuery"),
      iqiMap: document.getElementById("iqiMap"),
      iiqQuery: document.getElementById("iiqQuery"),
      iiqMap: document.getElementById("iiqMap"),
      deltaMap: document.getElementById("deltaMap"),
      iqiSurface: document.getElementById("iqiSurface"),
      iiqSurface: document.getElementById("iiqSurface"),
      deltaSurface: document.getElementById("deltaSurface"),
    }};
    const ctx = Object.fromEntries(Object.entries(canvases).map(([key, canvas]) => [key, canvas.getContext("2d")]));
    window.__CROSS_IMAGE_FLOW_CASES = window.__CROSS_IMAGE_FLOW_CASES || {{}};
    const caseCache = window.__CROSS_IMAGE_FLOW_CASES;
    const images = new Map();
    const MODE_IQI = INDEX.mode_order?.[0] || "image_text_image";
    const MODE_IIQ = INDEX.mode_order?.[1] || "image_image_text";
    let selectedCaseId = INDEX.paired_case_ids[0];
    let selectedQuery = 0;
    let currentPair = null;

    function ensureImage(src) {{
      if (!images.has(src)) {{
        const image = new Image();
        image.src = src;
        images.set(src, image);
      }}
      return images.get(src);
    }}

    function ensureImageLoaded(src) {{
      const image = ensureImage(src);
      if (image.complete && image.naturalWidth > 0) return Promise.resolve(image);
      return new Promise((resolve, reject) => {{
        image.onload = () => resolve(image);
        image.onerror = reject;
      }});
    }}

    function loadCase(caseId) {{
      if (caseCache[caseId]) return Promise.resolve(caseCache[caseId]);
      return new Promise((resolve, reject) => {{
        const script = document.createElement("script");
        script.src = INDEX.case_index[caseId].bundle_relpath;
        script.onload = () => resolve(caseCache[caseId]);
        script.onerror = () => reject(new Error(`Failed to load case bundle: ${{caseId}}`));
        document.body.appendChild(script);
      }});
    }}

    function pairRecord() {{
      return currentPair;
    }}

    function modeRecord(modeName) {{
      return pairRecord().modes[modeName];
    }}

    function matrixFor(modeName) {{
      return modeRecord(modeName).normalized_q_to_k;
    }}

    function gridDims(record, which) {{
      const shape = record.viewer_grid_shape?.[which] || record.raw_grid_shape?.[which] || null;
      if (shape) return {{llm_grid_h: shape.h, llm_grid_w: shape.w}};
      const tables = which === "image2" ? record.image2_patch_boxes : record.image1_patch_boxes;
      let maxRow = 0;
      let maxCol = 0;
      for (const patch of tables) {{
        maxRow = Math.max(maxRow, patch.row);
        maxCol = Math.max(maxCol, patch.col);
      }}
      return {{llm_grid_h: maxRow + 1, llm_grid_w: maxCol + 1}};
    }}

    function queryIndices(record) {{
      if (scopeSelect.value === "target_mean") {{
        return record.target_query_patch_indices.length ? record.target_query_patch_indices : [selectedQuery];
      }}
      if (scopeSelect.value === "full_mean") {{
        return record.image2_patch_boxes.map((_, idx) => idx);
      }}
      return [selectedQuery];
    }}

    function queryLabel(record) {{
      if (scopeSelect.value === "target_mean") {{
        return `target-box mean (${{queryIndices(record).length}} patches)`;
      }}
      if (scopeSelect.value === "full_mean") {{
        return `full-image mean (${{record.image2_patch_boxes.length}} patches)`;
      }}
      const patch = record.image2_patch_boxes[selectedQuery];
      return `patch r${{patch.row}}, c${{patch.col}}`;
    }}

    function aggregateRow(record) {{
      const matrix = record.normalized_q_to_k;
      const rows = queryIndices(record).map((idx) => matrix[idx]);
      const acc = new Array(rows[0].length).fill(0);
      for (const row of rows) {{
        for (let i = 0; i < row.length; i++) {{
          acc[i] += row[i];
        }}
      }}
      return acc.map((value) => value / rows.length);
    }}

    function normalize(row) {{
      const sum = row.reduce((acc, value) => acc + value, 0);
      if (sum <= 0) return row.map(() => 0);
      return row.map((value) => value / sum);
    }}

    function jsd(rowA, rowB) {{
      const p = normalize(rowA);
      const q = normalize(rowB);
      const m = p.map((value, idx) => 0.5 * (value + q[idx]));
      const kl = (a, b) => a.reduce((acc, value, idx) => {{
        if (value <= 0 || b[idx] <= 0) return acc;
        return acc + value * Math.log2(value / b[idx]);
      }}, 0);
      return 0.5 * kl(p, m) + 0.5 * kl(q, m);
    }}

    function contourIndices(row, massTarget) {{
      const ranked = row.map((value, idx) => [value, idx]).sort((a, b) => b[0] - a[0]);
      const keep = new Set();
      let acc = 0;
      for (const [value, idx] of ranked) {{
        keep.add(idx);
        acc += value;
        if (acc >= massTarget) break;
      }}
      return keep;
    }}

    function drawTargetBox(c, record) {{
      if (targetSelect.value !== "on") return;
      const box = record.target_box_xyxy || record.target_box || {{}};
      let x1; let y1; let x2; let y2;
      if (Array.isArray(box)) {{
        [x1, y1, x2, y2] = box;
      }} else if (box.x1 !== undefined) {{
        x1 = box.x1; y1 = box.y1; x2 = box.x2; y2 = box.y2;
      }} else if (box.x !== undefined) {{
        x1 = box.x; y1 = box.y; x2 = box.x + box.w; y2 = box.y + box.h;
      }} else {{
        return;
      }}
      c.strokeStyle = "#0f7c66";
      c.lineWidth = Math.max(2, c.canvas.width / 220);
      c.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }}

    function drawQueryPatch(c, grid, row, col) {{
      const cellW = c.canvas.width / grid.llm_grid_w;
      const cellH = c.canvas.height / grid.llm_grid_h;
      c.fillStyle = "rgba(178,78,51,0.22)";
      c.fillRect(col * cellW, row * cellH, cellW, cellH);
      c.strokeStyle = "#b24e33";
      c.lineWidth = Math.max(2, c.canvas.width / 250);
      c.strokeRect(col * cellW, row * cellH, cellW, cellH);
    }}

    function drawQuery(record, canvas, modeName) {{
      const image = ensureImage(record.image_relpath);
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const c = canvas.getContext("2d");
      c.clearRect(0, 0, canvas.width, canvas.height);
      c.drawImage(image, 0, 0);
      const grid = gridDims(record, "image2");
      const cellW = canvas.width / grid.llm_grid_w;
      const cellH = canvas.height / grid.llm_grid_h;
      c.strokeStyle = "rgba(31,28,25,0.14)";
      c.lineWidth = 1;
      for (let r = 0; r <= grid.llm_grid_h; r++) {{
        c.beginPath();
        c.moveTo(0, r * cellH);
        c.lineTo(canvas.width, r * cellH);
        c.stroke();
      }}
      for (let col = 0; col <= grid.llm_grid_w; col++) {{
        c.beginPath();
        c.moveTo(col * cellW, 0);
        c.lineTo(col * cellW, canvas.height);
        c.stroke();
      }}
      drawTargetBox(c, record);
      if (scopeSelect.value === "patch") {{
        const patch = record.image2_patch_boxes[selectedQuery];
        drawQueryPatch(c, grid, patch.row, patch.col);
      }} else {{
        for (const idx of queryIndices(record)) {{
          const patch = record.image2_patch_boxes[idx];
          drawQueryPatch(c, grid, patch.row, patch.col);
        }}
      }}
      c.fillStyle = "rgba(15,124,102,0.78)";
      c.font = `bold ${{Math.max(14, canvas.height / 18)}}px Georgia, serif`;
      c.fillText(modeLabel(modeName), 18, 34);
    }}

    function heatColor(value, maxAbs, signed, alpha) {{
      const t = Math.max(0, Math.min(1, Math.abs(value) / Math.max(maxAbs, 1e-8)));
      if (signed) {{
        const tone = value >= 0 ? [178, 78, 51] : [76, 114, 176];
        return `rgba(${{tone[0]}}, ${{tone[1]}}, ${{tone[2]}}, ${{0.10 + alpha * t}})`;
      }}
      return `rgba(15,124,102,${{0.10 + alpha * t}})`;
    }}

    function modeLabel(modeName) {{
      if (modeName === MODE_IQI) return "IQI";
      if (modeName === MODE_IIQ) return "IIQ";
      return modeName.replaceAll("_", " ").toUpperCase();
    }}

    function drawHeatmap(record, canvas, row, options = {{signed: false, title: ""}}) {{
      const image = ensureImage(record.image_relpath);
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const c = canvas.getContext("2d");
      c.clearRect(0, 0, canvas.width, canvas.height);
      c.drawImage(image, 0, 0);
      const grid = gridDims(record, "image1");
      const cellW = canvas.width / grid.llm_grid_w;
      const cellH = canvas.height / grid.llm_grid_h;
      const maxAbs = options.signed ? Math.max(...row.map((v) => Math.abs(v)), 1e-8) : Math.max(...row, 1e-8);
      const alpha = Number(alphaSelect.value);
      for (let i = 0; i < row.length; i++) {{
        const patch = record.image1_patch_boxes[i];
        c.fillStyle = heatColor(row[i], maxAbs, options.signed, alpha);
        c.fillRect(patch.col * cellW, patch.row * cellH, cellW, cellH);
      }}
      if (!options.signed) {{
        const contour = contourIndices(row, Number(massSelect.value));
        c.strokeStyle = "#b24e33";
        c.lineWidth = Math.max(2, canvas.width / 250);
        for (const idx of contour) {{
          const patch = record.image1_patch_boxes[idx];
          c.strokeRect(patch.col * cellW, patch.row * cellH, cellW, cellH);
        }}
      }}
      drawTargetBox(c, record);
      if (options.title) {{
        c.fillStyle = "rgba(30,27,24,0.88)";
        c.font = `bold ${{Math.max(15, canvas.height / 18)}}px Georgia, serif`;
        c.fillText(options.title, 18, 34);
      }}
    }}

    function poolMatrix(matrix, outRows = 24, outCols = 24) {{
      const rows = matrix.length;
      const cols = matrix[0].length;
      const pooled = [];
      for (let r = 0; r < outRows; r++) {{
        const r0 = Math.floor(r * rows / outRows);
        const r1 = Math.max(r0 + 1, Math.floor((r + 1) * rows / outRows));
        const outRow = [];
        for (let c = 0; c < outCols; c++) {{
          const c0 = Math.floor(c * cols / outCols);
          const c1 = Math.max(c0 + 1, Math.floor((c + 1) * cols / outCols));
          let sum = 0;
          let count = 0;
          for (let i = r0; i < r1; i++) {{
            for (let j = c0; j < c1; j++) {{
              sum += matrix[i][j];
              count += 1;
            }}
          }}
          outRow.push(count ? sum / count : 0);
        }}
        pooled.push(outRow);
      }}
      return pooled;
    }}

    function surfaceColor(value, maxAbs, signed) {{
      if (signed) {{
        const t = Math.max(0, Math.min(1, Math.abs(value) / Math.max(maxAbs, 1e-8)));
        const alpha = 0.25 + 0.70 * t;
        if (value >= 0) return `rgba(178, 78, 51, ${{alpha}})`;
        return `rgba(76, 114, 176, ${{alpha}})`;
      }}
      const t = Math.max(0, Math.min(1, value / Math.max(maxAbs, 1e-8)));
      return `rgba(15, 124, 102, ${{0.20 + 0.72 * t}})`;
    }}

    function drawPolygon(c, points, fillStyle, strokeStyle) {{
      c.beginPath();
      c.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) c.lineTo(points[i].x, points[i].y);
      c.closePath();
      c.fillStyle = fillStyle;
      c.fill();
      c.strokeStyle = strokeStyle;
      c.lineWidth = 0.6;
      c.stroke();
    }}

    function drawSurface(canvas, matrix, options = {{signed: false, title: ""}}) {{
      const pooled = poolMatrix(matrix, 22, 22);
      const rows = pooled.length;
      const cols = pooled[0].length;
      const width = 940;
      const height = 560;
      canvas.width = width;
      canvas.height = height;
      const c = canvas.getContext("2d");
      c.clearRect(0, 0, width, height);
      const values = pooled.flat();
      const maxAbs = options.signed ? Math.max(...values.map((v) => Math.abs(v)), 1e-8) : Math.max(...values, 1e-8);
      c.fillStyle = "rgba(255,255,255,0.96)";
      c.fillRect(0, 0, width, height);
      const originX = width * 0.52;
      const originY = height * 0.22;
      const scale = Math.min(width, height) * 0.30;
      const zScale = height * 0.44;
      function project(x, y, z) {{
        return {{
          x: originX + (x - y) * scale * 1.05,
          y: originY + (x + y) * scale * 0.52 - z * zScale,
        }};
      }}
      const cells = [];
      for (let r = 0; r < rows - 1; r++) {{
        for (let cIdx = 0; cIdx < cols - 1; cIdx++) {{
          const v = 0.25 * (
            pooled[r][cIdx] + pooled[r + 1][cIdx] + pooled[r][cIdx + 1] + pooled[r + 1][cIdx + 1]
          );
          const z = options.signed ? v / Math.max(maxAbs, 1e-8) : v / Math.max(maxAbs, 1e-8);
          const x0 = cIdx / (cols - 1) - 0.5;
          const x1 = (cIdx + 1) / (cols - 1) - 0.5;
          const y0 = r / (rows - 1) - 0.5;
          const y1 = (r + 1) / (rows - 1) - 0.5;
          cells.push({{
            depth: r + cIdx,
            value: v,
            points: [
              project(x0, y0, z),
              project(x1, y0, z),
              project(x1, y1, z),
              project(x0, y1, z),
            ],
          }});
        }}
      }}
      cells.sort((a, b) => a.depth - b.depth);
      for (const cell of cells) {{
        drawPolygon(c, cell.points, surfaceColor(cell.value, maxAbs, options.signed), "rgba(30,27,24,0.10)");
      }}
      c.strokeStyle = "rgba(30,27,24,0.20)";
      c.lineWidth = 1.0;
      const axisA = project(-0.5, -0.5, 0);
      const axisB = project(0.5, -0.5, 0);
      const axisC = project(-0.5, 0.5, 0);
      c.beginPath();
      c.moveTo(axisA.x, axisA.y);
      c.lineTo(axisB.x, axisB.y);
      c.lineTo(axisC.x, axisC.y);
      c.stroke();
      if (options.title) {{
        c.fillStyle = "rgba(30,27,24,0.90)";
        c.font = `bold 18px Georgia, serif`;
        c.fillText(options.title, 18, 34);
      }}
      c.fillStyle = "rgba(114,105,95,0.92)";
      c.font = "13px Georgia, serif";
      c.fillText(options.signed ? "blue < 0   red > 0" : "teal intensity = pooled mass", 18, height - 16);
    }}

    function targetMass(record) {{
      const row = aggregateRow(record);
      return record.target_key_patch_indices.reduce((acc, idx) => acc + row[idx], 0);
    }}

    function diagMass(record) {{
      const row = aggregateRow(record);
      const ids = queryIndices(record);
      let total = 0;
      for (const idx of ids) {{
        const patch = record.image2_patch_boxes[idx];
        for (let keyIdx = 0; keyIdx < record.image1_patch_boxes.length; keyIdx++) {{
          const keyPatch = record.image1_patch_boxes[keyIdx];
          if (Math.abs(keyPatch.row - patch.row) <= 1 && Math.abs(keyPatch.col - patch.col) <= 1) {{
            total += row[keyIdx] / ids.length;
          }}
        }}
      }}
      return total;
    }}

    function render() {{
      const pair = pairRecord();
      const iqi = pair.modes[MODE_IQI] || pair.modes[Object.keys(pair.modes)[0]];
      const iiq = pair.modes[MODE_IIQ] || pair.modes[Object.keys(pair.modes)[1]];
      const iqiMatrix = matrixFor(MODE_IQI);
      const iiqMatrix = matrixFor(MODE_IIQ);
      document.getElementById("questionText").textContent = pair.question;
      document.getElementById("answerText").textContent = pair.target_label || "-";
      document.getElementById("datasetText").textContent = pair.source?.dataset || "-";
      document.getElementById("baseText").textContent = pair.base_id || pair.id || "-";
      document.getElementById("kindText").textContent = pair.group_type || "-";
      document.getElementById("modePairText").textContent = `${{modeLabel(MODE_IQI)}} / ${{modeLabel(MODE_IIQ)}}`;
      drawQuery(iqi, canvases.iqiQuery, MODE_IQI);
      drawQuery(iiq, canvases.iiqQuery, MODE_IIQ);

      const iqiRow = aggregateRow(iqi);
      const iiqRow = aggregateRow(iiq);
      const deltaRow = iqiRow.map((value, idx) => value - iiqRow[idx]);
      drawHeatmap(iqi, canvases.iqiMap, iqiRow, {{signed: false, title: "IQI map"}});
      drawHeatmap(iiq, canvases.iiqMap, iiqRow, {{signed: false, title: "IIQ map"}});
      drawHeatmap(iiq, canvases.deltaMap, deltaRow, {{signed: true, title: "IQI - IIQ delta"}});

      drawSurface(canvases.iqiSurface, iqiMatrix, {{signed: false, title: "IQI surface"}});
      drawSurface(canvases.iiqSurface, iiqMatrix, {{signed: false, title: "IIQ surface"}});
      drawSurface(canvases.deltaSurface, iqiMatrix.map((row, rIdx) => row.map((value, cIdx) => value - iiqMatrix[rIdx][cIdx])), {{signed: true, title: "Delta surface"}});

      document.getElementById("metricScope").textContent = queryLabel(iqi);
      document.getElementById("metricIQITarget").textContent = targetMass(iqi).toFixed(3);
      document.getElementById("metricIIQTarget").textContent = targetMass(iiq).toFixed(3);
      document.getElementById("metricDeltaTarget").textContent = (targetMass(iqi) - targetMass(iiq)).toFixed(3);
      document.getElementById("metricIQIDiag").textContent = diagMass(iqi).toFixed(3);
      document.getElementById("metricIIQDiag").textContent = diagMass(iiq).toFixed(3);
      document.getElementById("metricDeltaDiag").textContent = (diagMass(iqi) - diagMass(iiq)).toFixed(3);
      document.getElementById("metricJSD").textContent = jsd(iqiRow, iiqRow).toFixed(3);
    }}

    function bindClicks(canvas, modeName) {{
      canvas.addEventListener("click", (event) => {{
        const record = modeRecord(modeName);
        const rect = canvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) * canvas.width / rect.width;
        const y = (event.clientY - rect.top) * canvas.height / rect.height;
        const grid = gridDims(record, "image2");
        const cellW = canvas.width / grid.llm_grid_w;
        const cellH = canvas.height / grid.llm_grid_h;
        const col = Math.min(grid.llm_grid_w - 1, Math.max(0, Math.floor(x / cellW)));
        const row = Math.min(grid.llm_grid_h - 1, Math.max(0, Math.floor(y / cellH)));
        selectedQuery = row * grid.llm_grid_w + col;
        scopeSelect.value = "patch";
        render();
      }});
    }}

    async function refreshCase(caseId) {{
      selectedCaseId = caseId;
      currentPair = await loadCase(caseId);
      const iqi = currentPair.modes[MODE_IQI] || currentPair.modes[Object.keys(currentPair.modes)[0]];
      const iiq = currentPair.modes[MODE_IIQ] || currentPair.modes[Object.keys(currentPair.modes)[1]];
      await Promise.all([
        ensureImageLoaded(iqi.image_relpath),
        ensureImageLoaded(iiq.image_relpath),
      ]);
      selectedQuery = 0;
      render();
    }}

    for (const caseId of INDEX.paired_case_ids) {{
      const meta = INDEX.case_index[caseId];
      const option = document.createElement("option");
      option.value = caseId;
      option.textContent = `${{caseId}} | ${{meta.kind}}`;
      caseSelect.appendChild(option);
    }}

    caseSelect.addEventListener("change", () => refreshCase(caseSelect.value));
    scopeSelect.addEventListener("change", render);
    massSelect.addEventListener("change", render);
    alphaSelect.addEventListener("change", render);
    targetSelect.addEventListener("change", render);
    bindClicks(canvases.iqiQuery, MODE_IQI);
    bindClicks(canvases.iiqQuery, MODE_IIQ);
    refreshCase(selectedCaseId);
  </script>
</body>
</html>
"""


def main() -> int:
    args = build_parser().parse_args()
    summary_path = Path(args.summary).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    data_dir = output_dir / "data"
    image_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, dict[str, Any]] = {}
    case_index: dict[str, dict[str, Any]] = {}

    items = summary.get("items") or summary.get("cases") or []
    by_case_id: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        case_id = item.get("id") or item.get("case_id")
        if not case_id:
            continue
        by_case_id.setdefault(case_id, []).append(item)

    for case_id, records in by_case_id.items():
        payload, image_path = grouped_case_payload(
            summary_path=summary_path,
            case_id=case_id,
            records=records,
            decimals=args.float_decimals,
        )
        target_image = image_dir / image_path.name
        if not target_image.exists():
            shutil.copy2(image_path, target_image)
        grouped[case_id] = payload

    paired_case_ids = sorted(
        [case_id for case_id, pair in grouped.items() if MODE_IQI in pair["modes"] and MODE_IIQ in pair["modes"]]
    )
    for case_id in paired_case_ids:
        pair = grouped[case_id]
        bundle_relpath = Path("data") / case_bundle_filename(case_id)
        case_index[case_id] = {
            "bundle_relpath": str(bundle_relpath),
            "base_id": pair["base_id"],
            "question_id": pair["question_id"],
            "question": pair["question"],
            "answer": pair["target_label"],
            "dataset": pair["source"].get("dataset", ""),
            "kind": pair["group_type"],
            "mode_pair": f"{mode_label(MODE_IQI)} / {mode_label(MODE_IIQ)}",
        }
        bundle_text = (
            "window.__CROSS_IMAGE_FLOW_CASES = window.__CROSS_IMAGE_FLOW_CASES || {};\n"
            f"window.__CROSS_IMAGE_FLOW_CASES[{json.dumps(case_id)}] = "
            f"{json.dumps(pair, ensure_ascii=False, separators=(',', ':'))};\n"
        )
        (output_dir / bundle_relpath).write_text(bundle_text, encoding="utf-8")

    index_payload = {
        "paired_case_ids": paired_case_ids,
        "case_index": case_index,
        "mode_order": [MODE_IQI, MODE_IIQ],
        "mode_labels": {MODE_IQI: mode_label(MODE_IQI), MODE_IIQ: mode_label(MODE_IIQ)},
    }
    (output_dir / "index.html").write_text(build_html(index_payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
