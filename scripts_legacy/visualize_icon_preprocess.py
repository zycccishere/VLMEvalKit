#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class CropDebug:
    applied: bool
    reason: str
    bg_gray: float
    bbox: tuple[int, int, int, int] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize icon crop + scale pipeline.")
    parser.add_argument("--icon-root", type=str, default="./icons-50/Icons-50")
    parser.add_argument("--output-dir", type=str, default="./exp_debug/icon_preprocess_debug_v1")
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260301)
    parser.add_argument("--trim-threshold", type=int, default=20)
    parser.add_argument("--icon-scale", type=float, default=0.82)
    parser.add_argument("--slot-width", type=int, default=120)
    parser.add_argument("--slot-height", type=int, default=220)
    parser.add_argument("--max-per-label", type=int, default=1, help="Max sampled icons per label.")
    parser.add_argument(
        "--image-prefix",
        type=str,
        default="all",
        choices=["all", "underscore", "tilde"],
        help="Filter by filename prefix: '_' for underscore, '~' for tilde.",
    )
    return parser.parse_args()


def corners_mean_rgb(icon: Image.Image) -> tuple[float, float, float]:
    px = icon.load()
    w, h = icon.size
    pts = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    rs, gs, bs = 0.0, 0.0, 0.0
    for x, y in pts:
        r, g, b = px[x, y]
        rs += r
        gs += g
        bs += b
    return rs / 4.0, gs / 4.0, bs / 4.0


def crop_light_border_debug(icon: Image.Image, threshold: int) -> tuple[Image.Image, CropDebug]:
    w, h = icon.size
    if w < 4 or h < 4:
        return icon, CropDebug(False, "too_small", 0.0, None)

    cr, cg, cb = corners_mean_rgb(icon)
    bg = (cr + cg + cb) / 3.0
    if min(cr, cg, cb) < 220:
        return icon, CropDebug(False, "dark_or_nonlight_corners", bg, None)

    gray = icon.convert("L")
    px = gray.load()
    x_min, y_min = w, h
    x_max, y_max = -1, -1
    for y in range(h):
        for x in range(w):
            v = px[x, y]
            if (bg - v) > threshold:
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y

    if x_max < 0 or y_max < 0:
        return icon, CropDebug(False, "no_foreground_found", bg, None)

    pad = 2
    x0 = max(0, x_min - pad)
    y0 = max(0, y_min - pad)
    x1 = min(w, x_max + 1 + pad)
    y1 = min(h, y_max + 1 + pad)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return icon, CropDebug(False, "bbox_too_small", bg, (x0, y0, x1, y1))
    return icon.crop((x0, y0, x1, y1)), CropDebug(True, "ok", bg, (x0, y0, x1, y1))


def render_slot(icon: Image.Image, slot_w: int, slot_h: int, icon_scale: float) -> Image.Image:
    cell = Image.new("RGB", (slot_w, slot_h), color=(248, 248, 248))
    draw = ImageDraw.Draw(cell)
    draw.rectangle((0, 0, slot_w - 1, slot_h - 1), outline=(220, 224, 235), width=1)
    tgt_w = max(4, int((slot_w - 14) * icon_scale))
    tgt_h = max(4, int((slot_h - 14) * icon_scale))
    fit = ImageOps.contain(icon, (tgt_w, tgt_h))
    px = (slot_w - fit.width) // 2
    py = (slot_h - fit.height) // 2
    cell.paste(fit, (px, py))
    return cell


def annotate_bbox(img: Image.Image, bbox: tuple[int, int, int, int] | None) -> Image.Image:
    out = img.copy()
    if bbox is not None:
        draw = ImageDraw.Draw(out)
        draw.rectangle(bbox, outline=(240, 30, 30), width=2)
    return out


def to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def collect_samples(
    icon_root: Path,
    num_samples: int,
    max_per_label: int,
    rng: random.Random,
    image_prefix: str,
) -> list[Path]:
    by_label: dict[str, list[Path]] = {}
    for d in sorted(icon_root.iterdir()):
        if not d.is_dir():
            continue
        imgs = []
        for p in d.iterdir():
            if p.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            if image_prefix == "underscore" and not p.name.startswith("_"):
                continue
            if image_prefix == "tilde" and not p.name.startswith("~"):
                continue
            imgs.append(p)
        imgs = sorted(imgs)
        if imgs:
            by_label[d.name] = imgs

    # Fairer sampling: cycle labels first, then fill remainder.
    selected: list[Path] = []
    labels = list(by_label.keys())
    rng.shuffle(labels)
    for label in labels:
        picks = by_label[label]
        rng.shuffle(picks)
        selected.extend(picks[:max_per_label])
        if len(selected) >= num_samples:
            return selected[:num_samples]

    all_files = [p for arr in by_label.values() for p in arr]
    rng.shuffle(all_files)
    for p in all_files:
        if len(selected) >= num_samples:
            break
        if p not in selected:
            selected.append(p)
    return selected[:num_samples]


def main() -> None:
    args = parse_args()
    if not (0.3 <= args.icon_scale <= 0.98):
        raise ValueError("--icon-scale should be in [0.3, 0.98]")

    icon_root = Path(args.icon_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    samples = collect_samples(
        icon_root,
        args.num_samples,
        args.max_per_label,
        rng,
        args.image_prefix,
    )

    cards_html: list[str] = []
    for idx, path in enumerate(samples, start=1):
        label = path.parent.name
        with Image.open(path) as im:
            raw = im.convert("RGB")
        cropped, dbg = crop_light_border_debug(raw, threshold=args.trim_threshold)
        with_box = annotate_bbox(raw, dbg.bbox)
        slot = render_slot(cropped, args.slot_width, args.slot_height, args.icon_scale)

        cards_html.append(
            f"""
            <section class="card">
              <div class="head">#{idx} · label={label} · file={path.name}</div>
              <div class="meta">
                applied={dbg.applied} · reason={dbg.reason} · bg_gray={dbg.bg_gray:.1f} · bbox={dbg.bbox}
              </div>
              <div class="grid">
                <div><div class="cap">1) raw</div><img src="{to_data_uri(raw)}" /></div>
                <div><div class="cap">2) raw + bbox</div><img src="{to_data_uri(with_box)}" /></div>
                <div><div class="cap">3) cropped</div><img src="{to_data_uri(cropped)}" /></div>
                <div><div class="cap">4) slot after scaling</div><img src="{to_data_uri(slot)}" /></div>
              </div>
            </section>
            """
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Icon Preprocess Debug</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#f5f7fb; color:#1f2430; margin:0; }}
    .wrap {{ max-width:1400px; margin:0 auto; padding:20px; }}
    .title {{ font-size:24px; font-weight:700; margin-bottom:10px; }}
    .sub {{ color:#5f6878; margin-bottom:14px; }}
    .card {{ background:#fff; border-radius:12px; padding:12px; margin:12px 0; box-shadow:0 2px 12px rgba(0,0,0,0.06); }}
    .head {{ font-weight:700; margin-bottom:6px; }}
    .meta {{ font-size:12px; color:#6f7890; margin-bottom:10px; }}
    .grid {{ display:grid; grid-template-columns:repeat(4, minmax(120px, 1fr)); gap:10px; }}
    .cap {{ font-size:12px; color:#5c6678; margin:4px 0; }}
    img {{ width:100%; height:auto; border:1px solid #e2e8f0; border-radius:8px; background:#fff; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns:repeat(2, minmax(120px,1fr)); }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="title">Icon Crop + Scale Debug</div>
    <div class="sub">
      trim_threshold={args.trim_threshold}, icon_scale={args.icon_scale}, slot=({args.slot_width}x{args.slot_height}), samples={len(samples)}, prefix={args.image_prefix}
    </div>
    {''.join(cards_html)}
  </div>
</body>
</html>
"""
    out_html = out_dir / "icon_preprocess_debug.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"[DONE] {out_html}")


if __name__ == "__main__":
    main()
