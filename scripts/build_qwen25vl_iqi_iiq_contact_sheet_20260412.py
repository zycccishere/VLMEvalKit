#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a contact sheet for IQI>IIQ seed candidates.")
    parser.add_argument("--selection-csv", required=True)
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--thumb-width", type=int, default=320)
    parser.add_argument("--thumb-height", type=int, default=240)
    parser.add_argument("--cols", type=int, default=4)
    return parser


def truncate(text: str, limit: int = 64) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def main() -> int:
    args = build_parser().parse_args()
    df = pd.read_csv(args.selection_csv)
    thumb_w = args.thumb_width
    thumb_h = args.thumb_height
    cols = args.cols
    rows = (len(df) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 88)), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, row in enumerate(df.itertuples()):
        image = Image.open(row.image_path).convert("RGB")
        scale = min((thumb_w - 10) / image.width, (thumb_h - 10) / image.height)
        resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
        tile = Image.new("RGB", (thumb_w, thumb_h), "#f5f1ea")
        offset_x = (thumb_w - resized.width) // 2
        offset_y = (thumb_h - resized.height) // 2
        tile.paste(resized, (offset_x, offset_y))

        gx = (idx % cols) * thumb_w
        gy = (idx // cols) * (thumb_h + 88)
        canvas.paste(tile, (gx, gy))
        draw.text((gx + 6, gy + thumb_h + 6), f"{row.base_id} | {row.source_dataset}", fill="black")
        draw.text((gx + 6, gy + thumb_h + 24), f"group={row.group} score={row.selection_score:.3f}", fill="#444444")
        draw.text((gx + 6, gy + thumb_h + 42), truncate(row.question, 56), fill="#444444")
        draw.text((gx + 6, gy + thumb_h + 60), f"IQI={truncate(row.iqi_prediction, 20)} | IIQ={truncate(row.iiq_prediction, 20)}", fill="#666666")

    output_image = Path(args.output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_image, quality=90)
    print(output_image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
