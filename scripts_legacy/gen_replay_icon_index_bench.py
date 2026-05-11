#!/usr/bin/env python3
"""
Build replay-oriented A/B datasets using real icons from icons-50.

Setting A:
  [image icons in one row] + [ask nth icon by direction]
Setting B:
  [image icons in one row] + [names list] + [mapping explanation] + [ask name->icon]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from urllib import request

import pandas as pd
from PIL import Image, ImageOps


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


DEFAULT_FIRST_NAMES_URL = (
    "https://raw.githubusercontent.com/smashew/NameDatabases/master/NamesDatabases/first%20names/us.txt"
)
DEFAULT_LAST_NAMES_URL = (
    "https://raw.githubusercontent.com/smashew/NameDatabases/master/NamesDatabases/surnames/us.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate replay benchmark from icons-50.")
    parser.add_argument(
        "--icon-root",
        type=str,
        default="./icons-50/Icons-50",
        help="Root directory containing class subfolders of icon images.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./exp_debug/replay_icon_index_data_v1",
        help="Output root directory.",
    )
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        default="ReplayIconIndex",
        help="Output TSV prefix. Will create <prefix>A.tsv and <prefix>B.tsv.",
    )
    parser.add_argument("--num-scenes", type=int, default=300, help="Number of generated scenes.")
    parser.add_argument(
        "--qa-per-direction-per-scene",
        type=int,
        default=1,
        help="How many QA samples to generate for each direction on each scene.",
    )
    parser.add_argument("--min-icons", type=int, default=8, help="Minimum icons per scene.")
    parser.add_argument("--max-icons", type=int, default=16, help="Maximum icons per scene.")
    parser.add_argument(
        "--min-images-per-label",
        type=int,
        default=30,
        help="Exclude icon labels with too few source images.",
    )
    parser.add_argument(
        "--image-prefix",
        type=str,
        default="all",
        choices=["all", "underscore", "tilde"],
        help="Filter source icon files by filename prefix: '_' or '~'.",
    )
    parser.add_argument("--seed", type=int, default=20260301, help="Random seed.")
    parser.add_argument("--image-width", type=int, default=1500, help="Canvas width.")
    parser.add_argument("--image-height", type=int, default=340, help="Canvas height.")
    parser.add_argument(
        "--icon-scale",
        type=float,
        default=0.82,
        help="Target icon fill ratio inside each slot (0.4~0.95 recommended).",
    )
    parser.add_argument(
        "--trim-light-border",
        type=int,
        default=1,
        help="Whether to trim light background borders before resizing (1/0).",
    )
    parser.add_argument(
        "--trim-threshold",
        type=int,
        default=20,
        help="Light-border tolerance for trimming (higher trims more).",
    )
    parser.add_argument("--name-source", type=str, default="real", choices=["real", "token"])
    parser.add_argument("--first-names-url", type=str, default=DEFAULT_FIRST_NAMES_URL)
    parser.add_argument("--last-names-url", type=str, default=DEFAULT_LAST_NAMES_URL)
    parser.add_argument("--name-cache-dir", type=str, default="")
    return parser.parse_args()


def label_to_text(label: str) -> str:
    return label.replace("_", " ")


def normalize_name_token(raw: str) -> str:
    x = raw.strip()
    x = re.sub(r"[^A-Za-z' -]", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    if not x:
        return ""
    return " ".join(part.capitalize() for part in x.split(" "))


def fetch_to_cache(url: str, cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return cache_path
    with request.urlopen(url, timeout=30) as resp:
        cache_path.write_bytes(resp.read())
    return cache_path


def read_name_lines(path: Path) -> list[str]:
    out: list[str] = []
    seen = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        name = normalize_name_token(line)
        if len(name) < 2 or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def build_name_pool(
    size: int,
    rng: random.Random,
    name_source: str,
    cache_dir: Path,
    first_names_url: str,
    last_names_url: str,
) -> list[str]:
    if name_source == "token":
        return [f"name_{i:04d}" for i in range(1, size + 1)]

    last_path = fetch_to_cache(last_names_url, cache_dir / "last_names.txt")
    last_names = read_name_lines(last_path)
    if len(last_names) < 100:
        raise ValueError("Downloaded surname list is too small.")
    pool = list(dict.fromkeys(last_names))
    rng.shuffle(pool)
    return pool


def collect_icon_pool(
    icon_root: Path,
    min_images_per_label: int,
    image_prefix: str,
) -> dict[str, list[Path]]:
    label_to_paths: dict[str, list[Path]] = {}
    for d in sorted(icon_root.iterdir()):
        if not d.is_dir():
            continue
        images = []
        for p in d.iterdir():
            if p.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            if image_prefix == "underscore" and not p.name.startswith("_"):
                continue
            if image_prefix == "tilde" and not p.name.startswith("~"):
                continue
            images.append(p)
        images = sorted(images)
        if len(images) >= min_images_per_label:
            label_to_paths[d.name] = images
    return label_to_paths


def render_icon_scene(
    icon_paths_lr: list[Path],
    out_path: Path,
    width: int,
    height: int,
    icon_scale: float,
    trim_light_border: bool,
    trim_threshold: int,
) -> None:
    canvas = Image.new("RGB", (width, height), color=(248, 248, 248))
    n = len(icon_paths_lr)
    margin_x = max(24, width // 50)
    margin_y = max(24, height // 9)
    gap = width * 0.007
    usable_w = width - 2 * margin_x - gap * (n - 1)
    slot_w = max(1, int(usable_w / n))
    slot_h = height - 2 * margin_y

    for i, path in enumerate(icon_paths_lr):
        x0 = int(margin_x + i * (slot_w + gap))
        y0 = int(margin_y)
        x1 = x0 + slot_w
        y1 = y0 + slot_h
        # cell border
        for x in range(x0, x1):
            canvas.putpixel((x, y0), (220, 224, 235))
            canvas.putpixel((x, y1 - 1), (220, 224, 235))
        for y in range(y0, y1):
            canvas.putpixel((x0, y), (220, 224, 235))
            canvas.putpixel((x1 - 1, y), (220, 224, 235))

        with Image.open(path) as icon:
            icon = preprocess_icon(
                icon.convert("RGB"),
                trim_light_border=trim_light_border,
                trim_threshold=trim_threshold,
            )
            tgt_w = max(4, int((slot_w - 14) * icon_scale))
            tgt_h = max(4, int((slot_h - 14) * icon_scale))
            icon = ImageOps.contain(icon, (tgt_w, tgt_h))
            px = x0 + (slot_w - icon.width) // 2
            py = y0 + (slot_h - icon.height) // 2
            canvas.paste(icon, (px, py))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def pick_direction(rng: random.Random) -> str:
    return "left_to_right" if rng.random() < 0.5 else "right_to_left"


def seq_by_direction(seq_lr: list[str], direction: str) -> list[str]:
    return seq_lr if direction == "left_to_right" else list(reversed(seq_lr))


def sample_target_positions(rng: random.Random, n_obj: int, k: int) -> list[int]:
    if k <= n_obj:
        return rng.sample(list(range(1, n_obj + 1)), k=k)
    picked = list(range(1, n_obj + 1))
    for _ in range(k - n_obj):
        picked.append(rng.randint(1, n_obj))
    rng.shuffle(picked)
    return picked


def choose_options(
    rng: random.Random,
    correct_label: str,
    label_space: list[str],
) -> tuple[list[str], str]:
    others = [x for x in label_space if x != correct_label]
    cands = rng.sample(others, 3) + [correct_label]
    rng.shuffle(cands)
    text = [label_to_text(x) for x in cands]
    answer = "ABCD"[cands.index(correct_label)]
    return text, answer


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


def crop_light_border(icon: Image.Image, threshold: int) -> Image.Image:
    """
    Trim near-uniform light borders to normalize visual icon scale.
    Only trims when corner color is very light; dark-background icons are kept.
    """
    w, h = icon.size
    if w < 4 or h < 4:
        return icon

    cr, cg, cb = corners_mean_rgb(icon)
    if min(cr, cg, cb) < 220:
        return icon

    bg = (cr + cg + cb) / 3.0
    gray = icon.convert("L")
    px = gray.load()

    x_min, y_min = w, h
    x_max, y_max = -1, -1
    # Identify non-background pixels.
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
        return icon

    # Keep tiny margin around content to avoid over-tight crop.
    pad = 2
    x0 = max(0, x_min - pad)
    y0 = max(0, y_min - pad)
    x1 = min(w, x_max + 1 + pad)
    y1 = min(h, y_max + 1 + pad)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return icon
    return icon.crop((x0, y0, x1, y1))


def preprocess_icon(icon: Image.Image, trim_light_border: bool, trim_threshold: int) -> Image.Image:
    if trim_light_border:
        icon = crop_light_border(icon, threshold=trim_threshold)
    return icon


def ordinal(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def question_a(target_pos: int, direction: str) -> str:
    d = "left to right" if direction == "left_to_right" else "right to left"
    return (
        "Icons are arranged in one row in the image.\n"
        f"What is the {ordinal(target_pos)} icon when counting from {d}?"
    )


def question_b(names: list[str], direction: str, target_name: str) -> str:
    d = "left to right" if direction == "left_to_right" else "right to left"
    return (
        f"Names: {', '.join(names)}\n"
        f"The names above are assigned one-to-one to icons in the image when counted from {d}.\n"
        f"Which icon is associated with {target_name}?"
    )


def main() -> None:
    args = parse_args()
    if args.min_icons < 3:
        raise ValueError("--min-icons must be >= 3")
    if args.max_icons < args.min_icons:
        raise ValueError("--max-icons must be >= --min-icons")
    if not (0.3 <= args.icon_scale <= 0.98):
        raise ValueError("--icon-scale should be in [0.3, 0.98]")
    if args.qa_per_direction_per_scene < 1:
        raise ValueError("--qa-per-direction-per-scene must be >= 1")

    rng = random.Random(args.seed)
    icon_root = Path(args.icon_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    img_dir = out_root / f"{args.dataset_prefix}_images"
    tsv_a = out_root / f"{args.dataset_prefix}A.tsv"
    tsv_b = out_root / f"{args.dataset_prefix}B.tsv"
    meta_path = out_root / f"{args.dataset_prefix}_meta.json"
    name_cache_dir = (
        Path(args.name_cache_dir).expanduser().resolve()
        if args.name_cache_dir
        else out_root / "_name_cache"
    )

    label_to_paths = collect_icon_pool(
        icon_root=icon_root,
        min_images_per_label=args.min_images_per_label,
        image_prefix=args.image_prefix,
    )
    labels = sorted(label_to_paths.keys())
    if len(labels) < args.max_icons:
        raise ValueError(
            f"Eligible labels too few ({len(labels)}), cannot sample max_icons={args.max_icons}. "
            "Try lowering --min-images-per-label or --max-icons."
        )

    name_pool = build_name_pool(
        size=max(5000, args.max_icons * args.num_scenes * 2),
        rng=rng,
        name_source=args.name_source,
        cache_dir=name_cache_dir,
        first_names_url=args.first_names_url,
        last_names_url=args.last_names_url,
    )
    if len(name_pool) < args.max_icons:
        raise ValueError(
            f"Name pool too small ({len(name_pool)}), cannot sample {args.max_icons} unique names per scene."
        )

    rows_a = []
    rows_b = []

    for idx in range(args.num_scenes):
        n = rng.randint(args.min_icons, args.max_icons)
        chosen_labels = rng.sample(labels, k=n)
        chosen_paths = [rng.choice(label_to_paths[label]) for label in chosen_labels]

        scene_id = f"scene_{idx:06d}"
        scene_img = (img_dir / f"{scene_id}.png").resolve()
        render_icon_scene(
            chosen_paths,
            scene_img,
            width=args.image_width,
            height=args.image_height,
            icon_scale=args.icon_scale,
            trim_light_border=bool(args.trim_light_border),
            trim_threshold=args.trim_threshold,
        )

        names = rng.sample(name_pool, k=n)
        for direction in ("left_to_right", "right_to_left"):
            seq = seq_by_direction(chosen_labels, direction)
            target_positions = sample_target_positions(rng, n_obj=n, k=args.qa_per_direction_per_scene)
            for qid, target_pos in enumerate(target_positions):
                tgt = seq[target_pos - 1]
                opt_a, ans_a = choose_options(rng, tgt, labels)
                rows_a.append(
                    {
                        "index": len(rows_a),
                        "image_path": str(scene_img),
                        "question": question_a(target_pos, direction),
                        "A": opt_a[0],
                        "B": opt_a[1],
                        "C": opt_a[2],
                        "D": opt_a[3],
                        "answer": ans_a,
                        "hide_options": 1,
                        "setting": "A",
                        "scene_id": scene_id,
                        "qa_id": qid,
                        "num_objects": n,
                        "direction": direction,
                        "target_pos": target_pos,
                        "target_object_id": tgt,
                    }
                )

                tgt_name = names[target_pos - 1]
                opt_b, ans_b = choose_options(rng, tgt, labels)
                rows_b.append(
                    {
                        "index": len(rows_b),
                        "image_path": str(scene_img),
                        "question": question_b(names, direction, tgt_name),
                        "A": opt_b[0],
                        "B": opt_b[1],
                        "C": opt_b[2],
                        "D": opt_b[3],
                        "answer": ans_b,
                        "hide_options": 1,
                        "setting": "B",
                        "scene_id": scene_id,
                        "qa_id": qid,
                        "num_objects": n,
                        "direction": direction,
                        "target_pos": target_pos,
                        "target_name": tgt_name,
                        "target_object_id": tgt,
                    }
                )

    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_a).to_csv(tsv_a, sep="\t", index=False)
    pd.DataFrame(rows_b).to_csv(tsv_b, sep="\t", index=False)

    meta = {
        "dataset_prefix": args.dataset_prefix,
        "icon_root": str(icon_root),
        "num_scenes": args.num_scenes,
        "qa_per_direction_per_scene": args.qa_per_direction_per_scene,
        "min_icons": args.min_icons,
        "max_icons": args.max_icons,
        "min_images_per_label": args.min_images_per_label,
        "image_prefix": args.image_prefix,
        "eligible_labels": len(labels),
        "name_source": args.name_source,
        "seed": args.seed,
        "image_width": args.image_width,
        "image_height": args.image_height,
        "icon_scale": args.icon_scale,
        "trim_light_border": bool(args.trim_light_border),
        "trim_threshold": args.trim_threshold,
        "tsv_a": str(tsv_a),
        "tsv_b": str(tsv_b),
        "image_dir": str(img_dir),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    print(f"[DONE] Setting A rows: {len(rows_a)}")
    print(f"[DONE] Setting B rows: {len(rows_b)}")


if __name__ == "__main__":
    main()
