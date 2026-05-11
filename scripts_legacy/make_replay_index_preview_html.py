#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import html
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTML preview for ReplayIndex A/B datasets."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Directory containing ReplayIndexA.tsv and ReplayIndexB.tsv.",
    )
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        default="ReplayIndex",
        help="Dataset prefix, e.g. ReplayIndex.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum paired scenes to show.",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default="",
        help="Output html path. Defaults to <data-root>/<prefix>_preview.html",
    )
    parser.add_argument(
        "--image-mode",
        type=str,
        default="relative",
        choices=["relative", "absolute", "inline"],
        help=(
            "How to reference images in html: "
            "relative (portable if keeping folder structure), "
            "absolute (file:///abs/path), "
            "inline (embed base64, fully self-contained html)."
        ),
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [row for row in reader]


def esc(v: str) -> str:
    return html.escape(v if v is not None else "")


def nl2br(v: str) -> str:
    return esc(v).replace("\n", "<br>")


def image_src(path_str: str, html_path: Path, mode: str) -> str:
    p = Path(path_str).expanduser().resolve()
    if mode == "absolute":
        try:
            return p.as_uri()
        except Exception:
            return "file://" + str(p)
    if mode == "inline":
        suffix = p.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        raw = p.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"
    # relative
    try:
        rel = p.relative_to(html_path.parent.resolve())
        return rel.as_posix()
    except ValueError:
        try:
            return p.as_uri()
        except Exception:
            return "file://" + str(p)


def render_question_block(row: dict[str, str], setting_name: str) -> str:
    answer = row.get("answer", "")
    options = [
        ("A", row.get("A", "")),
        ("B", row.get("B", "")),
        ("C", row.get("C", "")),
        ("D", row.get("D", "")),
    ]
    option_html = []
    for letter, text in options:
        cls = "opt"
        if letter == answer:
            cls += " correct"
        option_html.append(
            f"<div class='{cls}'><span class='letter'>{letter}.</span> {esc(text)}</div>"
        )

    meta = (
        f"direction={esc(row.get('direction', ''))}, "
        f"num_objects={esc(row.get('num_objects', ''))}, "
        f"target_pos={esc(row.get('target_pos', ''))}"
    )
    if setting_name == "B":
        meta += f", target_name={esc(row.get('target_name', ''))}"

    return f"""
    <div class="qa-card">
      <h3>Setting {setting_name}</h3>
      <div class="question">{nl2br(row.get("question", ""))}</div>
      <div class="options">{''.join(option_html)}</div>
      <div class="meta">{meta}</div>
    </div>
    """


def build_html(
    rows_a: list[dict[str, str]],
    rows_b: list[dict[str, str]],
    max_samples: int,
    dataset_prefix: str,
    html_path: Path,
    image_mode: str,
) -> str:
    by_scene_a = {r.get("scene_id", f"a_{i}"): r for i, r in enumerate(rows_a)}
    by_scene_b = {r.get("scene_id", f"b_{i}"): r for i, r in enumerate(rows_b)}
    scene_ids = sorted(set(by_scene_a).intersection(set(by_scene_b)))
    scene_ids = scene_ids[: max(0, max_samples)]

    cards = []
    for i, sid in enumerate(scene_ids, start=1):
        ra = by_scene_a[sid]
        rb = by_scene_b[sid]
        img = ra.get("image_path", "") or rb.get("image_path", "")
        cards.append(
            f"""
            <section class="scene-card">
              <div class="scene-head">
                <div class="scene-title">#{i} · {esc(sid)}</div>
                <div class="scene-sub">image: {esc(Path(img).name)}</div>
              </div>
              <div class="scene-body">
                <div class="img-wrap">
                  <img src="{esc(image_src(img, html_path=html_path, mode=image_mode))}" alt="{esc(sid)}" loading="lazy" />
                </div>
                <div class="qa-wrap">
                  {render_question_block(ra, "A")}
                  {render_question_block(rb, "B")}
                </div>
              </div>
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{esc(dataset_prefix)} Preview</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      background: #f5f7fb;
      color: #1f2430;
    }}
    .container {{
      max-width: 1300px;
      margin: 0 auto;
      padding: 24px;
    }}
    .title {{
      margin-bottom: 18px;
      font-size: 26px;
      font-weight: 700;
    }}
    .summary {{
      margin-bottom: 22px;
      color: #4d5666;
    }}
    .scene-card {{
      background: #fff;
      border-radius: 14px;
      box-shadow: 0 2px 14px rgba(0, 0, 0, 0.06);
      margin-bottom: 18px;
      overflow: hidden;
    }}
    .scene-head {{
      padding: 14px 16px;
      border-bottom: 1px solid #eef2f7;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .scene-title {{
      font-weight: 650;
    }}
    .scene-sub {{
      color: #687286;
      font-size: 13px;
    }}
    .scene-body {{
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 16px;
      padding: 16px;
    }}
    .img-wrap {{
      background: #f9fafc;
      border: 1px solid #eef2f7;
      border-radius: 10px;
      padding: 10px;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 220px;
    }}
    .img-wrap img {{
      width: 100%;
      height: auto;
      border-radius: 8px;
    }}
    .qa-wrap {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }}
    .qa-card {{
      border: 1px solid #e7ebf3;
      border-radius: 10px;
      padding: 12px;
      background: #fcfdff;
    }}
    .qa-card h3 {{
      margin: 0 0 8px 0;
      font-size: 15px;
    }}
    .question {{
      font-size: 13px;
      line-height: 1.5;
      color: #2c3445;
      margin-bottom: 8px;
      background: #f5f8ff;
      border-radius: 8px;
      padding: 8px;
      border: 1px solid #e7edff;
    }}
    .opt {{
      padding: 4px 8px;
      border-radius: 6px;
      margin-bottom: 4px;
      font-size: 13px;
      border: 1px solid transparent;
    }}
    .opt .letter {{
      font-weight: 600;
    }}
    .opt.correct {{
      background: #e9fff0;
      border-color: #b8efca;
    }}
    .meta {{
      margin-top: 8px;
      color: #6f7890;
      font-size: 12px;
    }}
    @media (max-width: 980px) {{
      .scene-body {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="title">{esc(dataset_prefix)} Dataset Preview</div>
    <div class="summary">
      Paired scenes shown: {len(scene_ids)} |
      Setting A rows: {len(rows_a)} |
      Setting B rows: {len(rows_b)} |
      Correct options are highlighted in green.
    </div>
    {''.join(cards)}
  </div>
</body>
</html>
"""
    return html_text


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    tsv_a = data_root / f"{args.dataset_prefix}A.tsv"
    tsv_b = data_root / f"{args.dataset_prefix}B.tsv"
    if not tsv_a.exists():
        raise FileNotFoundError(f"Missing file: {tsv_a}")
    if not tsv_b.exists():
        raise FileNotFoundError(f"Missing file: {tsv_b}")

    out_html = (
        Path(args.output_html).expanduser().resolve()
        if args.output_html
        else (data_root / f"{args.dataset_prefix}_preview.html")
    )

    rows_a = read_tsv(tsv_a)
    rows_b = read_tsv(tsv_b)
    html_text = build_html(
        rows_a=rows_a,
        rows_b=rows_b,
        max_samples=args.max_samples,
        dataset_prefix=args.dataset_prefix,
        html_path=out_html,
        image_mode=args.image_mode,
    )
    out_html.write_text(html_text, encoding="utf-8")
    print(f"[DONE] wrote preview html: {out_html}")


if __name__ == "__main__":
    main()
