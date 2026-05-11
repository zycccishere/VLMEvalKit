#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


SUBSET_NAMES = [
    "ReplayIconA_L2R",
    "ReplayIconA_R2L",
    "ReplayIconB_L2R",
    "ReplayIconB_R2L",
    "ReplayShapeA_L2R",
    "ReplayShapeA_R2L",
    "ReplayShapeB_L2R",
    "ReplayShapeB_R2L",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 8 replay subsets (A/B x icon/shape x L2R/R2L).")
    parser.add_argument("--output-root", type=str, default="./exp_debug/replay_8subsets_v1")
    parser.add_argument("--samples-per-subset", type=int, default=3000)
    parser.add_argument("--source-scenes", type=int, default=1500)
    parser.add_argument("--qa-per-direction-per-scene", type=int, default=2)
    parser.add_argument("--allow-replacement", type=int, default=1, help="Allow sampling with replacement if needed.")
    parser.add_argument("--seed", type=int, default=20260301)
    parser.add_argument("--icon-root", type=str, default="./icon-dataset/data")
    parser.add_argument(
        "--icon-prefix",
        type=str,
        default="underscore",
        choices=["all", "underscore", "tilde"],
        help="Use underscore by default (clean subset).",
    )
    parser.add_argument("--name-source", type=str, default="real", choices=["real", "token"])
    parser.add_argument("--name-cache-dir", type=str, default="./exp_debug/replay_index_data_v2/_name_cache")
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd))


def sample_direction_subset(
    df: pd.DataFrame,
    direction: str,
    sample_n: int,
    seed: int,
    allow_replacement: bool = True,
) -> pd.DataFrame:
    sub = df[df["direction"] == direction].copy()
    replace = False
    if len(sub) < sample_n:
        if not allow_replacement:
            raise ValueError(f"Not enough samples for direction={direction}: need={sample_n}, got={len(sub)}")
        replace = True
    sub = sub.sample(n=sample_n, random_state=seed, replace=replace).reset_index(drop=True)
    sub["index"] = list(range(len(sub)))
    if "question" in sub.columns:
        sub["question"] = sub["question"].astype(str)
        sub["question"] = sub["question"].str.replace("Answer with only one option letter.", "", regex=False)
        sub["question"] = sub["question"].str.replace("Answer with a single word or short phrase.", "", regex=False)
        sub["question"] = sub["question"].str.replace(
            "Answer the question using a single word or short phrase.", "", regex=False
        )
        sub["question"] = sub["question"].str.replace(
            "Answer directly with a single word or short phrase.", "", regex=False
        )
        sub["question"] = sub["question"].str.replace(
            "Do not output any explanation, derivation, words, or extra symbols.", "", regex=False
        )
        sub["question"] = sub["question"].map(lambda x: "\n".join([ln.rstrip() for ln in str(x).splitlines()]).strip())
    sub["hide_options"] = 1
    return sub


def dump_subset(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    gen_shape = root / "scripts" / "gen_replay_index_bench.py"
    gen_icon = root / "scripts" / "gen_replay_icon_index_bench.py"

    # 1) Build source pools (mixed directions)
    shape_pool = out_root / "_source_shape_pool"
    icon_pool = out_root / "_source_icon_pool"
    run_cmd(
        [
            py,
            str(gen_shape),
            "--output-root",
            str(shape_pool),
            "--dataset-prefix",
            "ReplayShapePool",
            "--num-scenes",
            str(args.source_scenes),
            "--qa-per-direction-per-scene",
            str(args.qa_per_direction_per_scene),
            "--min-objects",
            "10",
            "--max-objects",
            "14",
            "--seed",
            str(args.seed),
            "--name-source",
            args.name_source,
            "--name-cache-dir",
            str(Path(args.name_cache_dir).expanduser().resolve()),
        ],
        cwd=root,
    )
    run_cmd(
        [
            py,
            str(gen_icon),
            "--icon-root",
            str(Path(args.icon_root).expanduser().resolve()),
            "--output-root",
            str(icon_pool),
            "--dataset-prefix",
            "ReplayIconPool",
            "--num-scenes",
            str(args.source_scenes),
            "--qa-per-direction-per-scene",
            str(args.qa_per_direction_per_scene),
            "--min-icons",
            "10",
            "--max-icons",
            "14",
            "--min-images-per-label",
            "30",
            "--image-prefix",
            args.icon_prefix,
            "--seed",
            str(args.seed),
            "--name-source",
            args.name_source,
            "--name-cache-dir",
            str(Path(args.name_cache_dir).expanduser().resolve()),
        ],
        cwd=root,
    )

    # 2) Build 8 subsets
    shape_a = pd.read_csv(shape_pool / "ReplayShapePoolA.tsv", sep="\t")
    shape_b = pd.read_csv(shape_pool / "ReplayShapePoolB.tsv", sep="\t")
    icon_a = pd.read_csv(icon_pool / "ReplayIconPoolA.tsv", sep="\t")
    icon_b = pd.read_csv(icon_pool / "ReplayIconPoolB.tsv", sep="\t")

    rng_seed = args.seed
    mapping = [
        ("ReplayIconA_L2R", icon_a, "left_to_right", rng_seed + 1),
        ("ReplayIconA_R2L", icon_a, "right_to_left", rng_seed + 2),
        ("ReplayIconB_L2R", icon_b, "left_to_right", rng_seed + 3),
        ("ReplayIconB_R2L", icon_b, "right_to_left", rng_seed + 4),
        ("ReplayShapeA_L2R", shape_a, "left_to_right", rng_seed + 5),
        ("ReplayShapeA_R2L", shape_a, "right_to_left", rng_seed + 6),
        ("ReplayShapeB_L2R", shape_b, "left_to_right", rng_seed + 7),
        ("ReplayShapeB_R2L", shape_b, "right_to_left", rng_seed + 8),
    ]

    summary: dict[str, dict] = {}
    for name, df, direction, s in mapping:
        sampled = sample_direction_subset(
            df,
            direction=direction,
            sample_n=args.samples_per_subset,
            seed=s,
            allow_replacement=bool(args.allow_replacement),
        )
        out_path = out_root / f"{name}.tsv"
        dump_subset(sampled, out_path)
        summary[name] = {
            "rows": int(len(sampled)),
            "direction": direction,
            "file": str(out_path),
        }
        print(f"[DONE] {name}: {len(sampled)} rows -> {out_path}")

    # Write manifest and a ready-to-use DATALIST
    manifest = {
        "output_root": str(out_root),
        "samples_per_subset": args.samples_per_subset,
        "source_scenes": args.source_scenes,
        "qa_per_direction_per_scene": args.qa_per_direction_per_scene,
        "allow_replacement": bool(args.allow_replacement),
        "seed": args.seed,
        "icon_prefix": args.icon_prefix,
        "name_source": args.name_source,
        "subsets": summary,
        "datalist": SUBSET_NAMES,
    }
    (out_root / "replay_8subsets_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (out_root / "replay_8subsets_datalist.txt").write_text(
        " ".join(SUBSET_NAMES) + "\n",
        encoding="utf-8",
    )
    print("[DONE] Manifest:", out_root / "replay_8subsets_manifest.json")
    print("[DONE] DATALIST:", " ".join(SUBSET_NAMES))


if __name__ == "__main__":
    main()
