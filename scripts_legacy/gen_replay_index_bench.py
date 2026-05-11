#!/usr/bin/env python3
"""
Generate synthetic replay datasets for open-form answering.

Two settings are produced on the same image pool:
- Setting A: positional indexing (left-to-right / right-to-left).
- Setting B: name-to-object binding with explicit direction rule.

Output files:
- <output_root>/<dataset_prefix>A.tsv
- <output_root>/<dataset_prefix>B.tsv
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib import request

import pandas as pd
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class ObjectSpec:
    obj_id: str
    shape: str
    color: tuple[int, int, int]
    color_name: str

    @property
    def display_name(self) -> str:
        return f"{self.color_name} {self.shape}"


DEFAULT_FIRST_NAMES_URL = (
    "https://raw.githubusercontent.com/smashew/NameDatabases/master/NamesDatabases/first%20names/us.txt"
)
DEFAULT_LAST_NAMES_URL = (
    "https://raw.githubusercontent.com/smashew/NameDatabases/master/NamesDatabases/surnames/us.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate replay-oriented synthetic VLM benchmark.")
    parser.add_argument("--output-root", type=str, default="~/LMUData", help="Dataset output root.")
    parser.add_argument("--dataset-prefix", type=str, default="ReplayIndex", help="Output dataset prefix.")
    parser.add_argument("--num-scenes", type=int, default=500, help="Number of images/scenes.")
    parser.add_argument(
        "--qa-per-direction-per-scene",
        type=int,
        default=1,
        help="How many QA samples to generate for each direction on each scene.",
    )
    parser.add_argument("--min-objects", type=int, default=8, help="Minimum objects per image.")
    parser.add_argument("--max-objects", type=int, default=18, help="Maximum objects per image.")
    parser.add_argument("--seed", type=int, default=20260301, help="Random seed.")
    parser.add_argument("--image-width", type=int, default=1400, help="Canvas width.")
    parser.add_argument("--image-height", type=int, default=320, help="Canvas height.")
    parser.add_argument("--options", type=int, default=4, help="Number of hidden options for evaluation.")
    parser.add_argument(
        "--name-source",
        type=str,
        default="real",
        choices=["real", "token"],
        help="Name source: real (downloaded first/last names) or token (name_0001 style).",
    )
    parser.add_argument(
        "--first-names-url",
        type=str,
        default=DEFAULT_FIRST_NAMES_URL,
        help="URL for first name list (one name per line). Used when --name-source=real.",
    )
    parser.add_argument(
        "--last-names-url",
        type=str,
        default=DEFAULT_LAST_NAMES_URL,
        help="URL for last name list (one name per line). Used when --name-source=real.",
    )
    parser.add_argument(
        "--name-cache-dir",
        type=str,
        default="",
        help="Cache dir for downloaded name lists. Defaults to <output-root>/_name_cache.",
    )
    return parser.parse_args()


def build_object_vocab() -> list[ObjectSpec]:
    shapes = [
        "circle",
        "square",
        "triangle",
        "diamond",
        "pentagon",
        "hexagon",
        "plus",
        "cross",
    ]
    colors = [
        ("red", (220, 60, 60)),
        ("blue", (70, 110, 230)),
        ("green", (60, 170, 90)),
        ("orange", (245, 150, 50)),
        ("purple", (150, 90, 210)),
        ("teal", (50, 170, 170)),
    ]
    vocab: list[ObjectSpec] = []
    for color_name, color_value in colors:
        for shape in shapes:
            obj_id = f"{color_name}_{shape}"
            vocab.append(ObjectSpec(obj_id=obj_id, shape=shape, color=color_value, color_name=color_name))
    return vocab


def regular_polygon(cx: float, cy: float, radius: float, sides: int, rotate_deg: float = 0.0) -> list[tuple[float, float]]:
    points = []
    rotate = math.radians(rotate_deg)
    for i in range(sides):
        a = rotate + 2.0 * math.pi * i / sides
        points.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    return points


def draw_object(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], obj: ObjectSpec) -> None:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    r = 0.42 * min(x1 - x0, y1 - y0)
    fill = obj.color
    outline = (20, 20, 20)
    width = 3

    if obj.shape == "circle":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline, width=width)
    elif obj.shape == "square":
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline, width=width)
    elif obj.shape == "triangle":
        draw.polygon(regular_polygon(cx, cy, r, sides=3, rotate_deg=-90), fill=fill, outline=outline, width=width)
    elif obj.shape == "diamond":
        draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=fill, outline=outline, width=width)
    elif obj.shape == "pentagon":
        draw.polygon(regular_polygon(cx, cy, r, sides=5, rotate_deg=-90), fill=fill, outline=outline, width=width)
    elif obj.shape == "hexagon":
        draw.polygon(regular_polygon(cx, cy, r, sides=6, rotate_deg=0), fill=fill, outline=outline, width=width)
    elif obj.shape == "plus":
        arm = r * 0.42
        bar = r * 0.18
        rects = [
            (cx - bar, cy - arm, cx + bar, cy + arm),
            (cx - arm, cy - bar, cx + arm, cy + bar),
        ]
        for rect in rects:
            draw.rectangle(rect, fill=fill, outline=outline, width=width)
    elif obj.shape == "cross":
        draw.line((cx - r, cy - r, cx + r, cy + r), fill=fill, width=width + 3)
        draw.line((cx - r, cy + r, cx + r, cy - r), fill=fill, width=width + 3)
    else:
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline, width=width)


def render_scene_image(objects_lr: list[ObjectSpec], width: int, height: int, out_path: Path) -> None:
    img = Image.new("RGB", (width, height), color=(248, 248, 248))
    draw = ImageDraw.Draw(img)
    n = len(objects_lr)
    margin_x = max(30, width // 40)
    margin_y = max(30, height // 8)
    gap = width * 0.008
    usable_w = width - 2 * margin_x - gap * (n - 1)
    slot_w = usable_w / n
    y0 = margin_y
    y1 = height - margin_y

    for i, obj in enumerate(objects_lr):
        x0 = margin_x + i * (slot_w + gap)
        x1 = x0 + slot_w
        draw_object(draw, (x0, y0, x1, y1), obj)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def make_token_name_pool(size: int) -> list[str]:
    return [f"name_{i:04d}" for i in range(1, size + 1)]


def normalize_name_token(raw: str) -> str:
    x = raw.strip()
    x = re.sub(r"[^A-Za-z' -]", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    if not x:
        return ""
    # Keep simple title case for readability and consistency.
    return " ".join(part.capitalize() for part in x.split(" "))


def read_name_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[str] = []
    seen = set()
    for line in lines:
        name = normalize_name_token(line)
        if len(name) < 2:
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def fetch_to_cache(url: str, cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return cache_path
    with request.urlopen(url, timeout=30) as resp:
        raw = resp.read()
    cache_path.write_bytes(raw)
    return cache_path


def build_real_name_pool(
    size: int,
    rng: random.Random,
    cache_dir: Path,
    first_names_url: str,
    last_names_url: str,
) -> list[str]:
    first_path = fetch_to_cache(first_names_url, cache_dir / "first_names.txt")
    last_path = fetch_to_cache(last_names_url, cache_dir / "last_names.txt")
    first_names = read_name_lines(first_path)
    last_names = read_name_lines(last_path)
    if len(first_names) < 100 or len(last_names) < 100:
        raise ValueError(
            f"Name lists are too small: first={len(first_names)}, last={len(last_names)}"
        )

    # Build a large candidate pool first, then shuffle and take required size.
    target_candidates = max(size * 3, 20000)
    candidates = set()
    for _ in range(target_candidates):
        full = f"{rng.choice(first_names)} {rng.choice(last_names)}"
        candidates.add(full)
    if len(candidates) < size:
        # Deterministic fallback: expand by cartesian-like scan.
        for fn in first_names:
            for ln in last_names:
                candidates.add(f"{fn} {ln}")
                if len(candidates) >= size:
                    break
            if len(candidates) >= size:
                break
    if len(candidates) < size:
        raise ValueError(f"Failed to build enough unique full names: need={size}, got={len(candidates)}")

    pool = list(candidates)
    rng.shuffle(pool)
    return pool[:size]


def make_name_pool(
    size: int,
    rng: random.Random,
    name_source: str,
    cache_dir: Path,
    first_names_url: str,
    last_names_url: str,
) -> list[str]:
    if name_source == "token":
        return make_token_name_pool(size)
    return build_real_name_pool(
        size=size,
        rng=rng,
        cache_dir=cache_dir,
        first_names_url=first_names_url,
        last_names_url=last_names_url,
    )


def pick_direction(rng: random.Random) -> str:
    return "left_to_right" if rng.random() < 0.5 else "right_to_left"


def seq_by_direction(objects_lr: list[ObjectSpec], direction: str) -> list[ObjectSpec]:
    if direction == "left_to_right":
        return list(objects_lr)
    return list(reversed(objects_lr))


def sample_target_positions(rng: random.Random, n_obj: int, k: int) -> list[int]:
    if k <= n_obj:
        return rng.sample(list(range(1, n_obj + 1)), k=k)
    picked = list(range(1, n_obj + 1))
    for _ in range(k - n_obj):
        picked.append(rng.randint(1, n_obj))
    rng.shuffle(picked)
    return picked


def choose_mcq_options(
    rng: random.Random,
    correct_obj: ObjectSpec,
    vocab: list[ObjectSpec],
    num_options: int = 4,
) -> tuple[list[str], str]:
    if num_options < 2:
        raise ValueError("num_options must be >= 2")
    distractors = [x for x in vocab if x.obj_id != correct_obj.obj_id]
    sampled = rng.sample(distractors, k=num_options - 1)
    cand = sampled + [correct_obj]
    rng.shuffle(cand)
    option_text = [x.display_name for x in cand]
    answer_letter = "ABCD"[option_text.index(correct_obj.display_name)]
    return option_text, answer_letter


def build_question_a(target_pos: int, direction: str) -> str:
    def ordinal(n: int) -> str:
        if 10 <= (n % 100) <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    dir_text = "left to right" if direction == "left_to_right" else "right to left"
    return (
        "Objects are arranged in one row in the image.\n"
        f"What is the {ordinal(target_pos)} object when counting from {dir_text}?"
    )


def build_question_b(names: list[str], direction: str, target_name: str) -> str:
    dir_text = "left to right" if direction == "left_to_right" else "right to left"
    names_text = ", ".join(names)
    return (
        f"Names: {names_text}\n"
        f"The names above are assigned one-to-one to objects in the image when counted from {dir_text}.\n"
        f"Which object is associated with {target_name}?"
    )


def rows_to_tsv(rows: Iterable[dict], out_path: Path) -> None:
    df = pd.DataFrame(list(rows))
    df.to_csv(out_path, sep="\t", index=False)


def main() -> None:
    args = parse_args()
    if args.min_objects < 3:
        raise ValueError("--min-objects must be >= 3.")
    if args.max_objects < args.min_objects:
        raise ValueError("--max-objects must be >= --min-objects.")
    if args.options != 4:
        raise ValueError("This generator currently supports exactly 4 options (A/B/C/D).")
    if args.qa_per_direction_per_scene < 1:
        raise ValueError("--qa-per-direction-per-scene must be >= 1.")

    rng = random.Random(args.seed)
    out_root = Path(args.output_root).expanduser().resolve()
    name_cache_dir = (
        Path(args.name_cache_dir).expanduser().resolve()
        if args.name_cache_dir
        else (out_root / "_name_cache")
    )
    image_dir = out_root / f"{args.dataset_prefix}_images"
    tsv_a = out_root / f"{args.dataset_prefix}A.tsv"
    tsv_b = out_root / f"{args.dataset_prefix}B.tsv"
    meta_json = out_root / f"{args.dataset_prefix}_meta.json"

    vocab = build_object_vocab()
    if args.max_objects > len(vocab):
        raise ValueError(f"max objects ({args.max_objects}) exceeds vocab size ({len(vocab)}).")

    rows_a: list[dict] = []
    rows_b: list[dict] = []
    name_pool = make_name_pool(
        size=max(5000, args.max_objects * args.num_scenes * 2),
        rng=rng,
        name_source=args.name_source,
        cache_dir=name_cache_dir,
        first_names_url=args.first_names_url,
        last_names_url=args.last_names_url,
    )
    name_cursor = 0

    for scene_idx in range(args.num_scenes):
        n_obj = rng.randint(args.min_objects, args.max_objects)
        objects_lr = rng.sample(vocab, k=n_obj)

        scene_id = f"scene_{scene_idx:06d}"
        image_path = (image_dir / f"{scene_id}.png").resolve()
        render_scene_image(objects_lr, args.image_width, args.image_height, image_path)

        names = name_pool[name_cursor : name_cursor + n_obj]
        name_cursor += n_obj
        for direction in ("left_to_right", "right_to_left"):
            seq = seq_by_direction(objects_lr, direction)
            target_positions = sample_target_positions(rng, n_obj=n_obj, k=args.qa_per_direction_per_scene)
            for qid, target_pos in enumerate(target_positions):
                correct_obj = seq[target_pos - 1]
                options_a, answer_a = choose_mcq_options(rng, correct_obj, vocab, num_options=args.options)
                rows_a.append(
                    {
                        "index": len(rows_a),
                        "image_path": str(image_path),
                        "question": build_question_a(target_pos, direction),
                        "A": options_a[0],
                        "B": options_a[1],
                        "C": options_a[2],
                        "D": options_a[3],
                        "answer": answer_a,
                        "hide_options": 1,
                        "setting": "A",
                        "scene_id": scene_id,
                        "qa_id": qid,
                        "num_objects": n_obj,
                        "direction": direction,
                        "target_pos": target_pos,
                        "target_object_id": correct_obj.obj_id,
                    }
                )

                target_name = names[target_pos - 1]
                options_b, answer_b = choose_mcq_options(rng, correct_obj, vocab, num_options=args.options)
                rows_b.append(
                    {
                        "index": len(rows_b),
                        "image_path": str(image_path),
                        "question": build_question_b(names, direction, target_name),
                        "A": options_b[0],
                        "B": options_b[1],
                        "C": options_b[2],
                        "D": options_b[3],
                        "answer": answer_b,
                        "hide_options": 1,
                        "setting": "B",
                        "scene_id": scene_id,
                        "qa_id": qid,
                        "num_objects": n_obj,
                        "direction": direction,
                        "target_pos": target_pos,
                        "target_name": target_name,
                        "target_object_id": correct_obj.obj_id,
                    }
                )

    out_root.mkdir(parents=True, exist_ok=True)
    rows_to_tsv(rows_a, tsv_a)
    rows_to_tsv(rows_b, tsv_b)

    metadata = {
        "dataset_prefix": args.dataset_prefix,
        "num_scenes": args.num_scenes,
        "qa_per_direction_per_scene": args.qa_per_direction_per_scene,
        "min_objects": args.min_objects,
        "max_objects": args.max_objects,
        "seed": args.seed,
        "image_width": args.image_width,
        "image_height": args.image_height,
        "name_source": args.name_source,
        "name_cache_dir": str(name_cache_dir),
        "first_names_url": args.first_names_url,
        "last_names_url": args.last_names_url,
        "tsv_a": str(tsv_a),
        "tsv_b": str(tsv_b),
        "image_dir": str(image_dir),
    }
    meta_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    print(f"[DONE] Setting A rows: {len(rows_a)}")
    print(f"[DONE] Setting B rows: {len(rows_b)}")


if __name__ == "__main__":
    main()
