#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


MAPPING = [
    ("ReplayIconA_L2R", "_source_icon_pool/ReplayIconPoolA.tsv", "left_to_right", 1),
    ("ReplayIconA_R2L", "_source_icon_pool/ReplayIconPoolA.tsv", "right_to_left", 2),
    ("ReplayIconB_L2R", "_source_icon_pool/ReplayIconPoolB.tsv", "left_to_right", 3),
    ("ReplayIconB_R2L", "_source_icon_pool/ReplayIconPoolB.tsv", "right_to_left", 4),
    ("ReplayShapeA_L2R", "_source_shape_pool/ReplayShapePoolA.tsv", "left_to_right", 5),
    ("ReplayShapeA_R2L", "_source_shape_pool/ReplayShapePoolA.tsv", "right_to_left", 6),
    ("ReplayShapeB_L2R", "_source_shape_pool/ReplayShapePoolB.tsv", "left_to_right", 7),
    ("ReplayShapeB_R2L", "_source_shape_pool/ReplayShapePoolB.tsv", "right_to_left", 8),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample replay 8 subsets from existing source pools.")
    parser.add_argument("--output-root", type=str, default="./exp_debug/replay_8subsets_v1")
    parser.add_argument("--samples-per-subset", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260301)
    parser.add_argument("--allow-replacement", type=int, default=1, help="Allow sampling with replacement if needed.")
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def to_open_ended_question(question: str) -> str:
    q = str(question)
    q = q.replace("Answer with only one option letter.", "")
    q = q.replace("Answer with a single word or short phrase.", "")
    q = q.replace("Answer the question using a single word or short phrase.", "")
    q = q.replace("Answer directly with a single word or short phrase.", "")
    q = q.replace("Do not output any explanation, derivation, words, or extra symbols.", "")
    q = "\n".join([x.rstrip() for x in q.splitlines()]).strip()
    return q


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root).expanduser().resolve()
    manifest_path = out_root / "replay_8subsets_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary: dict[str, dict[str, str | int]] = {}

    for subset_name, source_rel, direction, seed_offset in MAPPING:
        src_path = out_root / source_rel
        rows = read_tsv(src_path)
        candidates = [r for r in rows if r.get("direction") == direction]
        rng = random.Random(args.seed + seed_offset)
        if len(candidates) >= args.samples_per_subset:
            sampled = rng.sample(candidates, args.samples_per_subset)
        else:
            if not bool(args.allow_replacement):
                raise ValueError(
                    f"Not enough rows for {subset_name}: need={args.samples_per_subset}, got={len(candidates)}"
                )
            sampled = [rng.choice(candidates) for _ in range(args.samples_per_subset)]
        for i, row in enumerate(sampled):
            row["index"] = str(i)
            row["question"] = to_open_ended_question(row.get("question", ""))
            row["hide_options"] = "1"

        out_path = out_root / f"{subset_name}.tsv"
        write_tsv(out_path, sampled)
        summary[subset_name] = {
            "rows": len(sampled),
            "direction": direction,
            "file": str(out_path),
        }
        print(f"[DONE] {subset_name}: {len(sampled)} rows -> {out_path}")

    manifest["samples_per_subset"] = args.samples_per_subset
    manifest["allow_replacement"] = bool(args.allow_replacement)
    manifest["seed"] = args.seed
    manifest["subsets"] = summary
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
