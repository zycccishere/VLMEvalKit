#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path

def image_to_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a single-file HTML viewer for IIT vs ITI cross-image flow probe outputs."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--probe-output-dir", required=True)
    parser.add_argument("--html-out", required=True)
    return parser


def load_cases(manifest_path: Path, probe_output_dir: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_path_by_id = {
        str(item["id"]): (manifest_path.parent / item["image_file"]).resolve()
        for item in manifest["items"]
    }
    summary = json.loads((probe_output_dir / "summary.json").read_text(encoding="utf-8"))
    cases_dir = probe_output_dir / "cases"
    data = {
        "manifest_id": summary["manifest_id"],
        "selected_layers": summary["selected_layers"],
        "items": [],
    }
    for meta in summary["items"]:
        case_path = cases_dir / f"{meta['id']}.json"
        case = json.loads(case_path.read_text(encoding="utf-8"))
        image_path = image_path_by_id.get(case["id"], Path(case["image_path"]))
        item = {
            "id": case["id"],
            "group_id": case["group_id"],
            "group_type": case["group_type"],
            "question": case["question"],
            "target_label": case["target_label"],
            "image_uri": image_to_data_uri(image_path),
            "modes": {},
        }
        for mode, payload in case["modes"].items():
            item["modes"][mode] = {
                "mode": mode,
                "prompt_text": payload["prompt_text"],
                "image_size": payload["image_size"],
                "target_box": payload["target_box"],
                "target_query_patch_indices": payload["target_query_patch_indices"],
                "target_key_patch_indices": payload["target_key_patch_indices"],
                "image1_patch_boxes": payload["image1_patch_boxes"],
                "image2_patch_boxes": payload["image2_patch_boxes"],
                "normalized_q_to_k": payload["normalized_q_to_k"],
                "summary_metrics": payload["summary_metrics"],
            }
        data["items"].append(item)
    return data


def render_html(data: dict[str, object]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen Cross-Image Flow Viewer</title>
  <style>
    :root {{
      --bg: #f4f0e8;
      --panel: #fffaf2;
      --ink: #18211f;
      --muted: #5a625c;
      --line: #d7ccb8;
      --accent: #8f3d2e;
      --accent-2: #225f75;
      --heat: #cc4b37;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      background:
        radial-gradient(circle at top left, rgba(143,61,46,0.10), transparent 34%),
        radial-gradient(circle at top right, rgba(34,95,117,0.10), transparent 28%),
        var(--bg);
      color: var(--ink);
    }}
    .page {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 28px 22px 36px;
    }}
    .header {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 20px;
      align-items: start;
      margin-bottom: 20px;
    }}
    .title-block {{
      background: rgba(255,250,242,0.82);
      border: 1px solid var(--line);
      padding: 18px 20px;
      border-radius: 18px;
      box-shadow: 0 12px 30px rgba(24,33,31,0.08);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0.01em;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }}
    .question {{
      margin-top: 16px;
      font-size: 19px;
      line-height: 1.45;
    }}
    .controls {{
      background: rgba(255,250,242,0.82);
      border: 1px solid var(--line);
      padding: 18px 20px;
      border-radius: 18px;
      box-shadow: 0 12px 30px rgba(24,33,31,0.08);
      display: grid;
      gap: 12px;
    }}
    .control-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }}
    label {{
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    select, button {{
      font: inherit;
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 9px 14px;
      background: white;
      color: var(--ink);
      cursor: pointer;
    }}
    button.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .metric-card {{
      background: rgba(255,250,242,0.82);
      border: 1px solid var(--line);
      padding: 14px 16px;
      border-radius: 16px;
      box-shadow: 0 8px 24px rgba(24,33,31,0.06);
    }}
    .metric-card h3 {{
      margin: 0 0 10px;
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }}
    .metric-line {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      font-size: 15px;
      margin: 4px 0;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .panel {{
      background: rgba(255,250,242,0.88);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      box-shadow: 0 12px 30px rgba(24,33,31,0.08);
    }}
    .panel h2 {{
      margin: 0 0 10px;
      font-size: 14px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .canvas-wrap {{
      display: flex;
      justify-content: center;
      align-items: center;
      background:
        linear-gradient(135deg, rgba(143,61,46,0.05), rgba(34,95,117,0.05));
      border-radius: 14px;
      overflow: hidden;
      min-height: 280px;
    }}
    canvas {{
      max-width: 100%;
      height: auto;
      cursor: crosshair;
      display: block;
    }}
    .legend {{
      margin-top: 18px;
      font-size: 13px;
      color: var(--muted);
      line-height: 1.6;
    }}
    .legend strong {{ color: var(--ink); }}
    @media (max-width: 1000px) {{
      .header, .metrics, .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div class="title-block">
        <h1>Qwen2.5-VL Cross-Image Flow</h1>
        <p class="subtitle">Top row: IIT (`image_image_text`). Bottom row: ITI (`image_text_image`). Click either left panel to select an <code>image2</code> patch. The right panel then shows which <code>image1</code> regions receive the highest normalized cross-image attention.</p>
        <div class="question" id="question"></div>
      </div>
      <div class="controls">
        <div>
          <label for="caseSelect">Case</label>
          <div class="control-row">
            <select id="caseSelect"></select>
          </div>
        </div>
        <div>
          <label>Aggregation</label>
          <div class="control-row">
            <button data-agg="patch" class="agg-btn active">Selected Patch</button>
            <button data-agg="target" class="agg-btn">Question Region Mean</button>
            <button data-agg="full" class="agg-btn">Full Image Mean</button>
          </div>
        </div>
      </div>
    </div>

    <div class="metrics">
      <div class="metric-card">
        <h3>Target Box Mass</h3>
        <div class="metric-line"><span>IIT full-query</span><span id="iitTargetMass"></span></div>
        <div class="metric-line"><span>ITI full-query</span><span id="itiTargetMass"></span></div>
        <div class="metric-line"><span>ITI - IIT</span><span id="deltaTargetMass"></span></div>
      </div>
      <div class="metric-card">
        <h3>Diagonal Mass</h3>
        <div class="metric-line"><span>IIT full-query</span><span id="iitDiagMass"></span></div>
        <div class="metric-line"><span>ITI full-query</span><span id="itiDiagMass"></span></div>
        <div class="metric-line"><span>ITI - IIT</span><span id="deltaDiagMass"></span></div>
      </div>
      <div class="metric-card">
        <h3>Cross-Image Mass</h3>
        <div class="metric-line"><span>IIT mean image1 mass</span><span id="iitCrossMass"></span></div>
        <div class="metric-line"><span>ITI mean image1 mass</span><span id="itiCrossMass"></span></div>
        <div class="metric-line"><span>Question target</span><span id="targetLabel"></span></div>
      </div>
    </div>

    <div class="grid">
      <section class="panel">
        <h2>IIT Image2</h2>
        <div class="canvas-wrap"><canvas id="iitLeft"></canvas></div>
      </section>
      <section class="panel">
        <h2>IIT Image1</h2>
        <div class="canvas-wrap"><canvas id="iitRight"></canvas></div>
      </section>
      <section class="panel">
        <h2>ITI Image2</h2>
        <div class="canvas-wrap"><canvas id="itiLeft"></canvas></div>
      </section>
      <section class="panel">
        <h2>ITI Image1</h2>
        <div class="canvas-wrap"><canvas id="itiRight"></canvas></div>
      </section>
    </div>

    <div class="legend">
      <strong>Red box</strong>: question target region. <strong>Blue patch</strong>: currently selected query patch on <code>image2</code>.
      <strong>Orange heat</strong>: normalized attention over <code>image1</code>. <strong>Dark contour</strong>: smallest patch set covering 50% of the displayed mass.
    </div>
  </div>

  <script>
    const DATA = {payload};
    const MODE_META = {{
      image_image_text: {{ name: "IIT" }},
      image_text_image: {{ name: "ITI" }},
    }};

    const state = {{
      caseId: DATA.items[0]?.id || null,
      agg: "patch",
      normPoint: {{ u: 0.5, v: 0.5 }},
    }};

    const imageCache = new Map();
    function getImage(uri) {{
      if (!imageCache.has(uri)) {{
        const img = new Image();
        img.src = uri;
        imageCache.set(uri, img);
      }}
      return imageCache.get(uri);
    }}

    function formatNum(value) {{
      return Number.isFinite(value) ? value.toFixed(3) : "nan";
    }}

    function nearestPatchIndex(patches, normPoint) {{
      let bestIdx = 0;
      let bestDist = Infinity;
      let width = 0;
      let height = 0;
      for (const patch of patches) {{
        width = Math.max(width, patch.x + patch.w);
        height = Math.max(height, patch.y + patch.h);
      }}
      for (let i = 0; i < patches.length; i++) {{
        const patch = patches[i];
        const u = (patch.x + patch.w / 2) / width;
        const v = (patch.y + patch.h / 2) / height;
        const dist = (u - normPoint.u) ** 2 + (v - normPoint.v) ** 2;
        if (dist < bestDist) {{
          bestDist = dist;
          bestIdx = i;
        }}
      }}
      return bestIdx;
    }}

    function getCurrentItem() {{
      return DATA.items.find((item) => item.id === state.caseId);
    }}

    function meanVectors(vectors) {{
      if (!vectors.length) return [];
      const out = new Array(vectors[0].length).fill(0);
      for (const vec of vectors) {{
        for (let i = 0; i < vec.length; i++) out[i] += vec[i];
      }}
      return out.map((v) => v / vectors.length);
    }}

    function getVectorForMode(modePayload) {{
      const matrix = modePayload.normalized_q_to_k;
      const queryIndices = (() => {{
        if (state.agg === "full") return matrix.map((_, idx) => idx);
        if (state.agg === "target" && modePayload.target_query_patch_indices.length) return modePayload.target_query_patch_indices;
        return [nearestPatchIndex(modePayload.image2_patch_boxes, state.normPoint)];
      }})();
      const rows = queryIndices
        .filter((idx) => idx >= 0 && idx < matrix.length)
        .map((idx) => matrix[idx]);
      return {{
        queryIndices,
        vector: meanVectors(rows),
      }};
    }}

    function topMassIndices(vector, threshold = 0.5) {{
      const pairs = vector.map((v, i) => [i, v]).sort((a, b) => b[1] - a[1]);
      const total = pairs.reduce((acc, [, v]) => acc + v, 0);
      if (!total) return [];
      let accum = 0;
      const out = [];
      for (const [idx, value] of pairs) {{
        accum += value;
        out.push(idx);
        if (accum / total >= threshold) break;
      }}
      return out;
    }}

    function drawImagePanel(canvas, item, modePayload, vector, queryIndices, isLeft) {{
      const img = getImage(item.image_uri);
      if (!img.complete) {{
        img.onload = () => render();
        return;
      }}
      const maxW = 520;
      const maxH = 420;
      const scale = Math.min(maxW / img.width, maxH / img.height);
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      const patches = isLeft ? modePayload.image2_patch_boxes : modePayload.image1_patch_boxes;
      const targetBox = modePayload.target_box;
      const sx = canvas.width / img.width;
      const sy = canvas.height / img.height;

      ctx.strokeStyle = "rgba(143,61,46,0.95)";
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 6]);
      ctx.strokeRect(targetBox.x * sx, targetBox.y * sy, targetBox.w * sx, targetBox.h * sy);
      ctx.setLineDash([]);

      if (isLeft) {{
        for (const idx of queryIndices) {{
          const patch = patches[idx];
          if (!patch) continue;
          ctx.fillStyle = "rgba(34,95,117,0.28)";
          ctx.strokeStyle = "rgba(34,95,117,0.95)";
          ctx.lineWidth = 2;
          ctx.fillRect(patch.x * sx, patch.y * sy, patch.w * sx, patch.h * sy);
          ctx.strokeRect(patch.x * sx, patch.y * sy, patch.w * sx, patch.h * sy);
        }}
      }} else {{
        const maxVal = Math.max(...vector, 1e-8);
        for (let i = 0; i < patches.length; i++) {{
          const patch = patches[i];
          const alpha = Math.min(0.82, Math.max(0, vector[i] / maxVal));
          if (alpha <= 0) continue;
          ctx.fillStyle = `rgba(204,75,55,${{alpha}})`;
          ctx.fillRect(patch.x * sx, patch.y * sy, patch.w * sx, patch.h * sy);
        }}
        const contour = topMassIndices(vector, 0.5);
        ctx.strokeStyle = "rgba(24,33,31,0.88)";
        ctx.lineWidth = 1.5;
        for (const idx of contour) {{
          const patch = patches[idx];
          if (!patch) continue;
          ctx.strokeRect(patch.x * sx, patch.y * sy, patch.w * sx, patch.h * sy);
        }}
      }}
    }}

    function updateMetrics(item) {{
      const iit = item.modes.image_image_text.summary_metrics;
      const iti = item.modes.image_text_image.summary_metrics;
      document.getElementById("iitTargetMass").textContent = formatNum(iit.mean_target_box_mass_full_query);
      document.getElementById("itiTargetMass").textContent = formatNum(iti.mean_target_box_mass_full_query);
      document.getElementById("deltaTargetMass").textContent = formatNum(iti.mean_target_box_mass_full_query - iit.mean_target_box_mass_full_query);
      document.getElementById("iitDiagMass").textContent = formatNum(iit.mean_diag_mass_full_query);
      document.getElementById("itiDiagMass").textContent = formatNum(iti.mean_diag_mass_full_query);
      document.getElementById("deltaDiagMass").textContent = formatNum(iti.mean_diag_mass_full_query - iit.mean_diag_mass_full_query);
      document.getElementById("iitCrossMass").textContent = formatNum(iit.mean_cross_image_mass);
      document.getElementById("itiCrossMass").textContent = formatNum(iti.mean_cross_image_mass);
      document.getElementById("targetLabel").textContent = item.target_label;
    }}

    function render() {{
      const item = getCurrentItem();
      if (!item) return;
      document.getElementById("question").textContent = item.question;
      updateMetrics(item);

      const iitPayload = item.modes.image_image_text;
      const itiPayload = item.modes.image_text_image;
      const iitVector = getVectorForMode(iitPayload);
      const itiVector = getVectorForMode(itiPayload);

      drawImagePanel(document.getElementById("iitLeft"), item, iitPayload, iitVector.vector, iitVector.queryIndices, true);
      drawImagePanel(document.getElementById("iitRight"), item, iitPayload, iitVector.vector, iitVector.queryIndices, false);
      drawImagePanel(document.getElementById("itiLeft"), item, itiPayload, itiVector.vector, itiVector.queryIndices, true);
      drawImagePanel(document.getElementById("itiRight"), item, itiPayload, itiVector.vector, itiVector.queryIndices, false);
    }}

    function setupSelector() {{
      const select = document.getElementById("caseSelect");
      for (const item of DATA.items) {{
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = `${{item.group_id}} | ${{item.id}}`;
        select.appendChild(option);
      }}
      select.value = state.caseId;
      select.addEventListener("change", (event) => {{
        state.caseId = event.target.value;
        state.normPoint = {{ u: 0.5, v: 0.5 }};
        render();
      }});
    }}

    function setupAggButtons() {{
      document.querySelectorAll(".agg-btn").forEach((button) => {{
        button.addEventListener("click", () => {{
          state.agg = button.dataset.agg;
          document.querySelectorAll(".agg-btn").forEach((btn) => btn.classList.toggle("active", btn === button));
          render();
        }});
      }});
    }}

    function setupCanvasClicks(canvasId, modeKey) {{
      const canvas = document.getElementById(canvasId);
      canvas.addEventListener("click", (event) => {{
        const item = getCurrentItem();
        if (!item) return;
        const modePayload = item.modes[modeKey];
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        state.normPoint = {{ u: x / rect.width, v: y / rect.height }};
        state.agg = "patch";
        document.querySelectorAll(".agg-btn").forEach((btn) => btn.classList.toggle("active", btn.dataset.agg === "patch"));
        render();
      }});
    }}

    setupSelector();
    setupAggButtons();
    setupCanvasClicks("iitLeft", "image_image_text");
    setupCanvasClicks("itiLeft", "image_text_image");
    render();
  </script>
</body>
</html>
"""


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest).resolve()
    probe_output_dir = Path(args.probe_output_dir).resolve()
    html_out = Path(args.html_out).resolve()
    data = load_cases(manifest_path, probe_output_dir)
    html_out.write_text(render_html(data), encoding="utf-8")
    print(json.dumps({"event": "html_written", "path": str(html_out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
