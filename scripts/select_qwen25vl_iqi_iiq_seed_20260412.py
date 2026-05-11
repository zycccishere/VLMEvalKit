#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select a reproducible 20-image IQI>IIQ seed shortlist.")
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reasoning-total", type=int, default=10)
    parser.add_argument("--non-reasoning-total", type=int, default=10)
    return parser


def score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    score = (
        out["grayscale_entropy"].fillna(0.0).astype(float)
        + 1.5 * out["edge_density"].fillna(0.0).astype(float)
        + 0.25 * out["pixel_std"].fillna(0.0).astype(float) / 255.0
    )
    out["selection_score"] = score
    out = out.sort_values(
        ["group", "source_dataset", "selection_score", "source_index"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    return out


def round_robin_unique(df: pd.DataFrame, total: int) -> pd.DataFrame:
    groups = {
        dataset: rows.reset_index(drop=True)
        for dataset, rows in df.groupby("source_dataset", sort=True)
    }
    cursors = {dataset: 0 for dataset in groups}
    used_images: set[str] = set()
    chosen_rows = []
    active = list(groups.keys())
    while len(chosen_rows) < total and active:
        next_active: list[str] = []
        for dataset in active:
            rows = groups[dataset]
            idx = cursors[dataset]
            picked = False
            while idx < len(rows):
                row = rows.iloc[idx]
                idx += 1
                image_path = str(row["image_path"])
                if image_path in used_images:
                    continue
                used_images.add(image_path)
                chosen_rows.append(row)
                picked = True
                break
            cursors[dataset] = idx
            if len(chosen_rows) >= total:
                break
            if idx < len(rows):
                next_active.append(dataset)
            elif picked:
                pass
        active = next_active
    if len(chosen_rows) < total:
        raise RuntimeError(f"Only selected {len(chosen_rows)} rows, below requested total {total}")
    return pd.DataFrame(chosen_rows).reset_index(drop=True)


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.candidates_csv)
    ranked = score_candidates(candidates)
    reasoning = ranked[ranked["group"] == "reasoning"].reset_index(drop=True)
    non_reasoning = ranked[ranked["group"] == "non_reasoning"].reset_index(drop=True)

    selected_reasoning = round_robin_unique(reasoning, args.reasoning_total)
    selected_non_reasoning = round_robin_unique(non_reasoning, args.non_reasoning_total)
    selected = pd.concat([selected_reasoning, selected_non_reasoning], ignore_index=True)
    selected = selected.sort_values(["group", "source_dataset", "selection_score"], ascending=[True, True, False]).reset_index(drop=True)
    selected.insert(0, "base_id", [f"{row.source_dataset.lower()}_{int(row.source_index)}" for row in selected.itertuples()])

    selected.to_csv(output_dir / "seed_selection.csv", index=False)
    ranked.to_csv(output_dir / "ranked_candidates.csv", index=False)

    summary = {
        "total_selected": int(len(selected)),
        "reasoning_selected": int((selected["group"] == "reasoning").sum()),
        "non_reasoning_selected": int((selected["group"] == "non_reasoning").sum()),
        "dataset_breakdown": (
            selected.groupby(["group", "source_dataset"]).size().reset_index(name="count").to_dict(orient="records")
        ),
    }
    (output_dir / "selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
