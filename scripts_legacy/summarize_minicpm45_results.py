#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET


KNOWN_DATASET_ORDER = [
    "AI2D_TEST",
    "DynaMath",
    "MathVision",
    "MathVista_MINI",
    "OCRBench",
    "SEEDBench2_Plus",
    "VisuLogic",
    "LogicVista",
    "VisualPuzzles",
]

MODEL_DIR_NAME = "MiniCPM-V-4_5-Replay"

SCORE_SUFFIXES = [
    ("_gpt-4o_score.csv", "score_csv"),
    ("_gpt-4-turbo_score.csv", "score_csv"),
    ("_score.csv", "score_csv"),
    ("_acc.csv", "acc_csv"),
    ("_score.json", "score_json"),
]

ARTIFACT_SUFFIXES = [
    "_gpt-4o_score.csv",
    "_gpt-4-turbo_score.csv",
    "_score.csv",
    "_acc.csv",
    "_score.json",
    "_gpt-4o.xlsx.bak",
    "_gpt-4-turbo.xlsx.bak",
    "_gpt-4o.xlsx",
    "_gpt-4-turbo.xlsx",
    "_gpt-4o_result.xlsx",
    "_gpt-4-turbo_result.xlsx",
    "_gpt-4o.pkl",
    "_gpt-4-turbo.pkl",
    "_gpt-4o_result.pkl",
    "_gpt-4-turbo_result.pkl",
    "_gpt-4o_result.json",
    "_answer_format_report.json",
    "_answer_format_failures.jsonl",
    ".xlsx",
    ".xlsx.bak",
    ".tsv",
    ".pkl",
    ".json",
]

NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-html-samples-per-pair", type=int, default=40)
    return parser.parse_args()


def split_task_name(task_name: str) -> Tuple[str, str]:
    parts = task_name.split("__")
    if len(parts) < 3:
        return task_name, task_name
    model = parts[0]
    setting = "__".join(parts[1:-1])
    return model, setting


def dataset_order(values: Iterable[str]) -> List[str]:
    values = list(values)
    known = [d for d in KNOWN_DATASET_ORDER if d in values]
    extras = sorted([d for d in values if d not in KNOWN_DATASET_ORDER])
    return known + extras


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def extract_dataset_name(file_name: str, model_dir_name: str) -> Optional[str]:
    prefix = f"{model_dir_name}_"
    if not file_name.startswith(prefix):
        return None
    rest = file_name[len(prefix):]
    for suffix in ARTIFACT_SUFFIXES:
        if rest.endswith(suffix):
            return rest[: -len(suffix)]
    return None


def parse_score_csv(path: Path) -> Tuple[Optional[str], Optional[float]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None, None
    if not rows:
        return None, None
    fieldnames = list(rows[0].keys())

    if "acc" in fieldnames:
        for key_col in ["Task&Skill", "Subject", "Setting", "split"]:
            if key_col in fieldnames:
                for row in rows:
                    if str(row.get(key_col, "")).strip() in {"Overall", "Average", "none"}:
                        try:
                            return "acc", float(row["acc"])
                        except Exception:
                            pass
        try:
            return "acc", float(rows[0]["acc"])
        except Exception:
            return None, None

    if "Overall" in fieldnames:
        try:
            return "Overall", float(rows[0]["Overall"])
        except Exception:
            return None, None

    for col in fieldnames:
        try:
            return col, float(rows[0][col])
        except Exception:
            continue
    return None, None


def parse_score_json(path: Path) -> Tuple[Optional[str], Optional[float]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, None

    for key in ["Final Score Norm", "Final Score", "Overall", "acc"]:
        if key in data and isinstance(data[key], (int, float)):
            return key, float(data[key])
    for key, value in data.items():
        if isinstance(value, (int, float)):
            return key, float(value)
    return None, None


def parse_score_file(path: Path, metric_type: str) -> Tuple[Optional[str], Optional[float]]:
    if metric_type in {"score_csv", "acc_csv"}:
        return parse_score_csv(path)
    if metric_type == "score_json":
        return parse_score_json(path)
    return None, None


def col_ref_to_idx(cell_ref: str) -> int:
    letters = []
    for ch in cell_ref:
        if ch.isalpha():
            letters.append(ch.upper())
        else:
            break
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return max(idx - 1, 0)


def read_xlsx_rows(path: Path) -> List[Dict[str, str]]:
    with zipfile.ZipFile(path, "r") as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("main:si", NS):
                parts = [t.text or "" for t in si.findall(".//main:t", NS)]
                shared_strings.append("".join(parts))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        first_sheet = workbook.find("main:sheets/main:sheet", NS)
        if first_sheet is None:
            return []
        rel_id = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")

        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels.findall("pkgrel:Relationship", NS):
            if rel.attrib.get("Id") == rel_id:
                target = rel.attrib.get("Target")
                break
        if not target:
            target = "worksheets/sheet1.xml"
        sheet_path = "xl/" + target.lstrip("/")
        root = ET.fromstring(zf.read(sheet_path))

        rows: List[List[str]] = []
        for row in root.findall("main:sheetData/main:row", NS):
            values: Dict[int, str] = {}
            max_idx = -1
            for cell in row.findall("main:c", NS):
                ref = cell.attrib.get("r", "")
                idx = col_ref_to_idx(ref)
                max_idx = max(max_idx, idx)
                cell_type = cell.attrib.get("t")
                value = ""
                if cell_type == "inlineStr":
                    node = cell.find("main:is/main:t", NS)
                    value = node.text if node is not None and node.text is not None else ""
                else:
                    node = cell.find("main:v", NS)
                    if node is not None and node.text is not None:
                        raw = node.text
                        if cell_type == "s":
                            try:
                                value = shared_strings[int(raw)]
                            except Exception:
                                value = raw
                        else:
                            value = raw
                values[idx] = value
            if max_idx < 0:
                rows.append([])
                continue
            rows.append([values.get(i, "") for i in range(max_idx + 1)])

    if not rows:
        return []
    header = [str(x).strip() for x in rows[0]]
    out: List[Dict[str, str]] = []
    for row in rows[1:]:
        item: Dict[str, str] = {}
        for i, key in enumerate(header):
            if not key:
                continue
            item[key] = row[i] if i < len(row) else ""
        if any(str(v).strip() for v in item.values()):
            out.append(item)
    return out


def percentile(values: List[float], q: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return float(values[0])
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    frac = pos - lo
    return float(xs[lo] * (1 - frac) + xs[hi] * frac)


def summarize_lengths(values: List[str], prefix: str) -> Dict[str, object]:
    lengths = [len(str(v)) for v in values if v is not None]
    nonempty = [v for v in values if str(v).strip()]
    nonempty_lengths = [len(str(v)) for v in nonempty]
    row: Dict[str, object] = {
        f"{prefix}_count": len(values),
        f"{prefix}_nonempty_count": len(nonempty),
        f"{prefix}_empty_count": len(values) - len(nonempty),
    }
    if not nonempty_lengths:
        row.update(
            {
                f"{prefix}_chars_min": "",
                f"{prefix}_chars_p50": "",
                f"{prefix}_chars_mean": "",
                f"{prefix}_chars_p90": "",
                f"{prefix}_chars_max": "",
            }
        )
        return row

    row.update(
        {
            f"{prefix}_chars_min": min(nonempty_lengths),
            f"{prefix}_chars_p50": percentile(nonempty_lengths, 0.5),
            f"{prefix}_chars_mean": statistics.mean(nonempty_lengths),
            f"{prefix}_chars_p90": percentile(nonempty_lengths, 0.9),
            f"{prefix}_chars_max": max(nonempty_lengths),
        }
    )
    return row


def collect_scores(run_root: Path) -> List[Dict]:
    best_scores: Dict[Tuple[str, str], Dict] = {}
    for task_dir in sorted(run_root.iterdir()):
        if not task_dir.is_dir():
            continue
        output_dir = task_dir / "output" / MODEL_DIR_NAME
        if not output_dir.is_dir():
            continue

        model, setting = split_task_name(task_dir.name)
        for file_path in sorted(output_dir.iterdir()):
            if not file_path.is_file():
                continue
            dataset = extract_dataset_name(file_path.name, MODEL_DIR_NAME)
            if not dataset:
                continue
            metric_type = None
            for suffix, mt in SCORE_SUFFIXES:
                if file_path.name.endswith(suffix):
                    metric_type = mt
                    break
            if metric_type is None:
                continue
            score_key, score_value = parse_score_file(file_path, metric_type)
            if score_value is None:
                continue
            key = (task_dir.name, dataset)
            best_scores[key] = {
                "pack": run_root.name,
                "task": task_dir.name,
                "model": model,
                "setting": setting,
                "dataset": dataset,
                "metric_type": metric_type,
                "score_key": score_key,
                "score_value": score_value,
                "abs_path": str(file_path),
            }
    return sorted(best_scores.values(), key=lambda x: (x["task"], x["dataset"]))


def write_pivot(scores: List[Dict], output_dir: Path) -> Path:
    pivot_dir = output_dir / "pivot_by_setting"
    pivot_dir.mkdir(parents=True, exist_ok=True)
    datasets = dataset_order(sorted({row["dataset"] for row in scores}))
    label_to_scores: Dict[str, Dict[str, float]] = {}
    for row in scores:
        label = f'{row["pack"]}|{row["model"]}|{row["setting"]}'
        label_to_scores.setdefault(label, {})
        label_to_scores[label][row["dataset"]] = row["score_value"]

    out_rows = []
    for label in sorted(label_to_scores.keys()):
        row = {"setting": label}
        for dataset in datasets:
            row[dataset] = label_to_scores[label].get(dataset, "")
        out_rows.append(row)

    out_path = pivot_dir / "ALL__dataset_x_setting.csv"
    write_csv(out_path, out_rows, ["setting"] + datasets)
    return out_path


def collect_prediction_length_stats(run_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for task_dir in sorted(run_root.iterdir()):
        if not task_dir.is_dir():
            continue
        output_dir = task_dir / "output" / MODEL_DIR_NAME
        if not output_dir.is_dir():
            continue
        model, setting = split_task_name(task_dir.name)

        for xlsx_path in sorted(output_dir.glob(f"{MODEL_DIR_NAME}_*.xlsx")):
            if xlsx_path.name.endswith("_gpt-4o.xlsx") or xlsx_path.name.endswith("_gpt-4o_result.xlsx"):
                continue
            dataset = extract_dataset_name(xlsx_path.name, MODEL_DIR_NAME)
            if not dataset:
                continue
            try:
                data_rows = read_xlsx_rows(xlsx_path)
            except Exception:
                continue

            predictions = [row.get("prediction", "") for row in data_rows]
            detailed_predictions = [row.get("detailed_prediction", "") for row in data_rows]
            stat_row: Dict[str, object] = {
                "pack": run_root.name,
                "task": task_dir.name,
                "model": model,
                "setting": setting,
                "dataset": dataset,
                "n_rows": len(data_rows),
                "xlsx_path": str(xlsx_path),
            }
            stat_row.update(summarize_lengths(predictions, "prediction"))
            stat_row.update(summarize_lengths(detailed_predictions, "detailed_prediction"))
            rows.append(stat_row)
    return rows


def collect_debug_records(run_root: Path, max_samples_per_pair: int) -> List[Dict]:
    records: List[Dict] = []
    per_pair_count: Dict[Tuple[str, str], int] = {}

    for task_dir in sorted(run_root.iterdir()):
        if not task_dir.is_dir():
            continue
        model, setting = split_task_name(task_dir.name)
        debug_root = task_dir / "output" / "_logs" / "minicpm_debug_io"
        if not debug_root.is_dir():
            continue
        for jsonl_path in sorted(debug_root.rglob("MiniCPM_V_4_5_Replay.jsonl")):
            try:
                lines = jsonl_path.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                dataset = str(item.get("dataset", "")).strip()
                if not dataset:
                    continue
                pair_key = (setting, dataset)
                if per_pair_count.get(pair_key, 0) >= max_samples_per_pair:
                    continue
                item["setting"] = setting
                item["task"] = task_dir.name
                item["model_name"] = model
                item["source_jsonl"] = str(jsonl_path)
                per_pair_count[pair_key] = per_pair_count.get(pair_key, 0) + 1
                records.append(item)
    return sorted(records, key=lambda x: (x["setting"], x["dataset"], int(x.get("call_index", 0))))


def esc(v: object) -> str:
    return html.escape("" if v is None else str(v))


def text_block(v: object) -> str:
    return esc(v).replace("\n", "<br>")


def first_image_path(input_message: object) -> str:
    if not isinstance(input_message, list):
        return ""
    for item in input_message:
        if isinstance(item, dict) and item.get("type") == "image":
            return str(item.get("value", ""))
    return ""


def render_input_message(input_message: object) -> str:
    if not isinstance(input_message, list):
        return f"<pre>{esc(json.dumps(input_message, ensure_ascii=False, indent=2))}</pre>"
    blocks = []
    for item in input_message:
        if not isinstance(item, dict):
            blocks.append(f"<div class='msg other'><pre>{esc(repr(item))}</pre></div>")
            continue
        item_type = str(item.get("type", "unknown"))
        value = item.get("value", "")
        if item_type == "text":
            blocks.append(f"<div class='msg text'><div class='label'>text</div><div>{text_block(value)}</div></div>")
        else:
            blocks.append(f"<div class='msg media'><div class='label'>{esc(item_type)}</div><div>{esc(value)}</div></div>")
    return "".join(blocks)


def render_debug_html(records: List[Dict], output_path: Path) -> None:
    settings = sorted({str(r.get("setting", "")) for r in records})
    datasets = dataset_order(sorted({str(r.get("dataset", "")) for r in records}))

    cards = []
    for idx, record in enumerate(records):
        setting = str(record.get("setting", ""))
        dataset = str(record.get("dataset", ""))
        call_index = record.get("call_index", "")
        image_path = first_image_path(record.get("input_message"))
        image_src = ""
        if image_path:
            try:
                image_src = Path(image_path).expanduser().resolve().as_uri()
            except Exception:
                image_src = "file://" + image_path
        cards.append(
            f"""
            <section class="card" data-setting="{esc(setting)}" data-dataset="{esc(dataset)}">
              <div class="head">
                <div class="title">{esc(setting)} | {esc(dataset)} | sample #{esc(call_index)}</div>
                <div class="meta">{esc(record.get("source_jsonl", ""))}</div>
              </div>
              <div class="body">
                <div class="image-col">
                  {f"<img src='{esc(image_src)}' loading='lazy' />" if image_src else "<div class='noimg'>No image</div>"}
                </div>
                <div class="content-col">
                  <div class="section">
                    <div class="section-title">Input Message</div>
                    {render_input_message(record.get("input_message"))}
                  </div>
                  <div class="section">
                    <div class="section-title">Raw Output</div>
                    <pre>{esc(record.get("raw_output", ""))}</pre>
                  </div>
                  <div class="section">
                    <div class="section-title">Final Output</div>
                    <pre>{esc(record.get("final_output", ""))}</pre>
                  </div>
                </div>
              </div>
            </section>
            """
        )

    settings_options = "".join([f"<option value='{esc(s)}'>{esc(s)}</option>" for s in settings])
    datasets_options = "".join([f"<option value='{esc(d)}'>{esc(d)}</option>" for d in datasets])

    html_text = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MiniCPM Debug IO Preview</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f1220;
      color: #eef2ff;
    }}
    .wrap {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 20px;
    }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
      background: rgba(15, 18, 32, 0.95);
      padding: 12px 0 16px;
      backdrop-filter: blur(8px);
    }}
    select {{
      background: #1a1f36;
      color: #eef2ff;
      border: 1px solid #394065;
      border-radius: 8px;
      padding: 8px 10px;
    }}
    .count {{
      color: #b7c0ea;
    }}
    .card {{
      background: #171b2e;
      border: 1px solid #2d3558;
      border-radius: 14px;
      margin: 0 0 18px 0;
      overflow: hidden;
    }}
    .head {{
      padding: 12px 16px;
      border-bottom: 1px solid #2d3558;
    }}
    .title {{
      font-weight: 700;
    }}
    .meta {{
      margin-top: 4px;
      font-size: 12px;
      color: #93a0d9;
      word-break: break-all;
    }}
    .body {{
      display: grid;
      grid-template-columns: 440px 1fr;
      gap: 16px;
      padding: 16px;
    }}
    .image-col img {{
      width: 100%;
      border-radius: 10px;
      background: #fff;
    }}
    .noimg {{
      padding: 24px;
      border: 1px dashed #49527f;
      border-radius: 10px;
      color: #8e98c8;
    }}
    .section {{
      margin-bottom: 14px;
    }}
    .section-title {{
      font-weight: 650;
      margin-bottom: 8px;
      color: #dbe2ff;
    }}
    .msg {{
      border-radius: 10px;
      padding: 10px 12px;
      margin-bottom: 8px;
      border: 1px solid #303960;
    }}
    .msg.text {{
      background: #12172b;
    }}
    .msg.media {{
      background: #1b2140;
    }}
    .label {{
      font-size: 12px;
      color: #8ea0f5;
      margin-bottom: 6px;
      text-transform: uppercase;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #101425;
      border: 1px solid #2d3558;
      border-radius: 10px;
      padding: 12px;
      color: #edf1ff;
    }}
    @media (max-width: 1100px) {{
      .body {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="toolbar">
      <label>Setting
        <select id="settingFilter">
          <option value="">全部</option>
          {settings_options}
        </select>
      </label>
      <label>Dataset
        <select id="datasetFilter">
          <option value="">全部</option>
          {datasets_options}
        </select>
      </label>
      <div class="count">样本数: <span id="visibleCount">{len(records)}</span> / {len(records)}</div>
    </div>
    {''.join(cards)}
  </div>
  <script>
    const settingFilter = document.getElementById('settingFilter');
    const datasetFilter = document.getElementById('datasetFilter');
    const cards = Array.from(document.querySelectorAll('.card'));
    const visibleCount = document.getElementById('visibleCount');
    function applyFilters() {{
      const setting = settingFilter.value;
      const dataset = datasetFilter.value;
      let count = 0;
      cards.forEach(card => {{
        const ok = (!setting || card.dataset.setting === setting) && (!dataset || card.dataset.dataset === dataset);
        card.style.display = ok ? '' : 'none';
        if (ok) count += 1;
      }});
      visibleCount.textContent = count;
    }}
    settingFilter.addEventListener('change', applyFilters);
    datasetFilter.addEventListener('change', applyFilters);
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = collect_scores(run_root)
    write_csv(
        output_dir / "existing_primary_scores.csv",
        scores,
        ["pack", "task", "model", "setting", "dataset", "metric_type", "score_key", "score_value", "abs_path"],
    )
    pivot_path = write_pivot(scores, output_dir)

    length_rows = collect_prediction_length_stats(run_root)
    write_csv(
        output_dir / "prediction_length_stats.csv",
        length_rows,
        [
            "pack",
            "task",
            "model",
            "setting",
            "dataset",
            "n_rows",
            "prediction_count",
            "prediction_nonempty_count",
            "prediction_empty_count",
            "prediction_chars_min",
            "prediction_chars_p50",
            "prediction_chars_mean",
            "prediction_chars_p90",
            "prediction_chars_max",
            "detailed_prediction_count",
            "detailed_prediction_nonempty_count",
            "detailed_prediction_empty_count",
            "detailed_prediction_chars_min",
            "detailed_prediction_chars_p50",
            "detailed_prediction_chars_mean",
            "detailed_prediction_chars_p90",
            "detailed_prediction_chars_max",
            "xlsx_path",
        ],
    )

    debug_records = collect_debug_records(run_root, args.max_html_samples_per_pair)
    write_csv(
        output_dir / "debug_io_inventory.csv",
        [
            {
                "task": r.get("task", ""),
                "model_name": r.get("model_name", ""),
                "setting": r.get("setting", ""),
                "dataset": r.get("dataset", ""),
                "call_index": r.get("call_index", ""),
                "source_jsonl": r.get("source_jsonl", ""),
            }
            for r in debug_records
        ],
        ["task", "model_name", "setting", "dataset", "call_index", "source_jsonl"],
    )
    render_debug_html(debug_records, output_dir / "minicpm_debug_io_preview.html")

    print(json.dumps(
        {
            "run_root": str(run_root),
            "output_dir": str(output_dir),
            "pivot_csv": str(pivot_path),
            "prediction_length_csv": str(output_dir / "prediction_length_stats.csv"),
            "debug_html": str(output_dir / "minicpm_debug_io_preview.html"),
            "score_rows": len(scores),
            "length_rows": len(length_rows),
            "debug_records": len(debug_records),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
