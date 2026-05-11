#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw


WORKSPACE = Path("/path/to/WorkHub")
RESOURCE_ROOT = WORKSPACE / "assets/topics/topic-image-replay/resources/cross_image_flow_controlled_set_20x2_v3_20260412"
SELECTION_CSV = (
    WORKSPACE
    / "assets/topics/topic-image-replay/resources/qwen25vl-iqi-iiq-candidates-20260412/seed_selection/seed_selection.csv"
)
ALT_PATH = RESOURCE_ROOT / "alternate_questions.json"
BOX_PATH = RESOURCE_ROOT / "box_annotations.json"
MANIFEST_SEED_PATH = RESOURCE_ROOT / "manifest_seed.json"
MANIFEST_PATH = RESOURCE_ROOT / "manifest.json"
SELECTION_MD_PATH = RESOURCE_ROOT / "selection.md"
PREVIEW_PATH = RESOURCE_ROOT / "box_preview_contact_sheet.jpg"

COLORS = {
    "orig": "#0f7c66",
    "alt": "#b24e33",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_box(image_size: tuple[int, int], box: dict[str, int], key: str) -> None:
    width, height = image_size
    x = int(box["x"])
    y = int(box["y"])
    w = int(box["w"])
    h = int(box["h"])
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(f"Invalid box {key}: {box}")
    if x + w > width or y + h > height:
        raise ValueError(f"Box {key} exceeds image bounds {image_size}: {box}")


def selection_note(row: dict[str, str]) -> str:
    dataset = row["source_dataset"]
    group = row["group"]
    if dataset == "AI2D_TEST":
        return "Labeled science diagram with two spatially separated label targets."
    if dataset == "OCRBench":
        return "OCR-heavy case with one benchmark query and one alternate local evidence query."
    if dataset == "DynaMath":
        return "Reasoning case with one global benchmark question and one localized supporting-region probe."
    if dataset == "LogicVista":
        return "Table/diagram reasoning case where the alternate question isolates a second region on the same image."
    if dataset == "MathVision":
        return "Math reasoning figure with one broad benchmark question and one tighter regional follow-up."
    if dataset == "VisualPuzzles":
        return "Analogy case where the benchmark target is the chosen candidate and the alternate target is a source-region cue."
    return f"{group} case with one benchmark question and one alternate same-image control question."


def answer_text(value: str) -> str:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value.replace("'", '"'))
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except Exception:
            return value
    return value


def build_manifest_seed(rows: list[dict[str, str]], alternates: dict[str, dict[str, str]]) -> dict:
    items: list[dict] = []
    for row in rows:
        base_id = row["base_id"]
        if base_id not in alternates:
            raise KeyError(f"Missing alternate question for {base_id}")
        preferred_name = f"{base_id}.jpg"
        if (RESOURCE_ROOT / "images" / preferred_name).exists():
            image_name = preferred_name
        else:
            image_name = Path(row["image_path"]).name
        items.append(
            {
                "id": base_id,
                "source_dataset": row["source_dataset"],
                "source_row_index": int(row["source_index"]),
                "group": row["group"],
                "image_file": f"images/{image_name}",
                "selection_note": selection_note(row),
                "source": {
                    "candidate_id": row["candidate_id"],
                    "iqi_prediction": row["iqi_prediction"],
                    "iiq_prediction": row["iiq_prediction"],
                    "selection_score": float(row["selection_score"]),
                },
                "questions": [
                    {
                        "id": "orig",
                        "kind": "benchmark",
                        "target_label": answer_text(row["answer"]),
                        "question": row["question"],
                    },
                    {
                        "id": "alt",
                        "kind": "alternate",
                        "target_label": alternates[base_id]["answer"],
                        "question": alternates[base_id]["question"],
                        "target_description": alternates[base_id]["target_description"],
                    },
                ],
            }
        )
    return {
        "id": "qwen25vl_iqi_iiq_controlled_set_20x2_v3_seed_20260412",
        "topic": "topic-image-replay",
        "version": "2026-04-12",
        "status": "box_pending",
        "description": (
            "Human-curated 20-image x 2-question controlled pack for Qwen2.5-VL-32B IQI vs IIQ cross-image flow analysis. "
            "Each image carries the original IQI-correct/IIQ-wrong benchmark question plus one controlled alternate question."
        ),
        "items": items,
    }


def build_runtime_manifest(seed: dict, box_annotations: dict[str, dict[str, dict[str, int]]]) -> list[dict]:
    manifest: list[dict] = []
    for item in seed["items"]:
        image_path = RESOURCE_ROOT / item["image_file"]
        image_size = Image.open(image_path).size
        item_boxes = box_annotations[item["id"]]
        questions_out: list[dict] = []
        for question in item["questions"]:
            qid = question["id"]
            if qid not in item_boxes:
                raise KeyError(f"Missing box for {item['id']}.{qid}")
            box = {k: int(v) for k, v in item_boxes[qid].items()}
            assert_box(image_size, box, f"{item['id']}.{qid}")
            questions_out.append(
                {
                    "id": qid,
                    "kind": question["kind"],
                    "question": question["question"],
                    "answer": question["target_label"],
                    "target_box_xyxy": [box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]],
                    "target_description": question.get("target_description", ""),
                }
            )
        manifest.append(
            {
                "id": item["id"],
                "image": item["image_file"],
                "source": {
                    "dataset": item["source_dataset"],
                    "row_index": int(item["source_row_index"]),
                    "group": item["group"],
                    "selection_note": item["selection_note"],
                    **item["source"],
                },
                "questions": questions_out,
            }
        )
    return manifest


def build_preview(seed: dict, box_annotations: dict[str, dict[str, dict[str, int]]]) -> None:
    thumb_w, thumb_h = 320, 220
    cols = 4
    rows = (len(seed["items"]) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 52)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, item in enumerate(seed["items"]):
        path = RESOURCE_ROOT / item["image_file"]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        scale = min((thumb_w - 8) / width, (thumb_h - 8) / height)
        resized = image.resize((int(width * scale), int(height * scale)))
        tile = Image.new("RGB", (thumb_w, thumb_h), "#f5f1ea")
        offset_x = (thumb_w - resized.width) // 2
        offset_y = (thumb_h - resized.height) // 2
        tile.paste(resized, (offset_x, offset_y))
        tile_draw = ImageDraw.Draw(tile)
        for question in item["questions"]:
            box = box_annotations[item["id"]][question["id"]]
            x = offset_x + int(box["x"] * scale)
            y = offset_y + int(box["y"] * scale)
            w = int(box["w"] * scale)
            h = int(box["h"] * scale)
            color = COLORS[question["id"]]
            tile_draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            tile_draw.text((x + 4, max(0, y - 14)), question["id"], fill=color)
        gx = (idx % cols) * thumb_w
        gy = (idx // cols) * (thumb_h + 52)
        canvas.paste(tile, (gx, gy))
        draw.text((gx + 6, gy + thumb_h + 6), item["id"], fill="black")
        draw.text((gx + 6, gy + thumb_h + 24), "orig=green, alt=red", fill="#444444")
    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(PREVIEW_PATH, quality=90)


def build_selection_md(seed: dict) -> str:
    lines: list[str] = []
    lines.append("# 20x2 Selection")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This pack refreshes the `IQI` vs `IIQ` controlled viewer with `20` base images and `1` alternate question per image.")
    lines.append("- Locked topology definitions:")
    lines.append("  - `IQI = image_text_image`")
    lines.append("  - `IIQ = image_image_text`")
    lines.append("- Selection rule:")
    lines.append("  - all base cases come from the `Qwen2.5-VL-32B` V1 root where `IQI` is correct and `IIQ` is wrong")
    lines.append("  - `10 reasoning + 10 non_reasoning`")
    lines.append("- Important annotation caveat:")
    lines.append("  - for some broad reasoning benchmark questions, the `orig` target box is a curator-chosen evidence region rather than a uniquely defined ground-truth box")
    lines.append("- Runtime files:")
    lines.append(f"  - question seed: `[file]:topic-image-replay/resources/{RESOURCE_ROOT.name}/manifest_seed.json`")
    lines.append(f"  - box annotations: `[file]:topic-image-replay/resources/{RESOURCE_ROOT.name}/box_annotations.json`")
    lines.append(f"  - runnable manifest: `[file]:topic-image-replay/resources/{RESOURCE_ROOT.name}/manifest.json`")
    lines.append(f"  - preview: `[file]:topic-image-replay/resources/{RESOURCE_ROOT.name}/box_preview_contact_sheet.jpg`")
    lines.append("")
    lines.append("## Selected Cases")
    lines.append("")
    for item in seed["items"]:
        lines.append(f"### {item['id']}")
        lines.append(f"- Dataset: `{item['source_dataset']}`")
        lines.append(f"- Group: `{item['group']}`")
        lines.append(f"- Note: {item['selection_note']}")
        lines.append(f"- `orig`: {item['questions'][0]['question']}")
        lines.append(f"- `alt`: {item['questions'][1]['question']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = load_rows(SELECTION_CSV)
    alternates = load_json(ALT_PATH)
    box_annotations = load_json(BOX_PATH)
    seed = build_manifest_seed(rows, alternates)
    manifest = build_runtime_manifest(seed, box_annotations)
    MANIFEST_SEED_PATH.write_text(json.dumps(seed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SELECTION_MD_PATH.write_text(build_selection_md(seed), encoding="utf-8")
    build_preview(seed, box_annotations)
    print(MANIFEST_SEED_PATH)
    print(MANIFEST_PATH)
    print(SELECTION_MD_PATH)
    print(PREVIEW_PATH)
    print(len(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
