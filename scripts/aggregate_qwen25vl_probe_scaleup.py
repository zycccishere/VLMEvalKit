#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATASET_NAME_MAP = {
    "dynamath": "DynaMath",
    "logicvista": "LogicVista",
    "seedbench": "SEEDBench2_Plus",
    "seedbench2_plus": "SEEDBench2_Plus",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate Qwen2.5-VL probe scale-up shards and render summary figures."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def concat_csvs(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def canonical_dataset_name(raw: str) -> str:
    key = raw.strip().lower()
    return DATASET_NAME_MAP.get(key, raw)


def concat_csvs_with_dataset(paths: list[Path], prefix: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["dataset"] = canonical_dataset_name(path.parent.parent.name.replace(prefix, ""))
        frame["job"] = path.parent.parent.name
        frame["shard"] = path.parent.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_prefill(prefill_df: pd.DataFrame, out_path: Path) -> None:
    if prefill_df.empty:
        return
    grouped = prefill_df.groupby("dataset", as_index=False)[["image1_mass", "image2_mass", "text_mass"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = {"image1_mass": "#4c72b0", "image2_mass": "#dd8452", "text_mass": "#55a868"}
    for axis, column in zip(axes, ["image1_mass", "image2_mass", "text_mass"]):
        axis.bar(grouped["dataset"], grouped[column], color=colors[column])
        axis.set_title(column.replace("_", " "))
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle("Prefill Last-Token Attention Mass", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_decode(decode_df: pd.DataFrame, out_path: Path) -> None:
    if decode_df.empty:
        return
    grouped = decode_df.groupby(["dataset", "step"], as_index=False)[["image1_mass", "image2_mass", "text_mass"]].mean()
    datasets = grouped["dataset"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for axis, dataset in zip(axes, datasets):
        sub = grouped[grouped["dataset"] == dataset]
        axis.plot(sub["step"], sub["image1_mass"], label="image1", color="#4c72b0")
        axis.plot(sub["step"], sub["image2_mass"], label="image2", color="#dd8452")
        axis.plot(sub["step"], sub["text_mass"], label="text", color="#55a868")
        axis.set_title(dataset)
        axis.set_xlabel("decode step")
        axis.set_ylabel("attention mass")
        axis.legend()
    fig.suptitle("Decode Attention Over Steps", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_alignment(alignment_df: pd.DataFrame, out_path: Path) -> None:
    if alignment_df.empty:
        return
    grouped = alignment_df.groupby(["dataset", "layer"], as_index=False).mean(numeric_only=True)
    datasets = grouped["dataset"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(datasets), 2, figsize=(12, 4 * len(datasets)))
    if len(datasets) == 1:
        axes = [axes]
    for row_axes, dataset in zip(axes, datasets):
        sub = grouped[grouped["dataset"] == dataset]
        row_axes[0].plot(sub["layer"], sub["image1_pairwise_cos"], label="image1")
        row_axes[0].plot(sub["layer"], sub["image2_pairwise_cos"], label="image2")
        row_axes[0].set_title(f"{dataset}: Pairwise Cosine To Text")
        row_axes[0].set_xlabel("layer")
        row_axes[0].set_ylabel("cosine")
        row_axes[0].legend()

        row_axes[1].plot(sub["layer"], sub["image1_text_subspace_ratio"], label="image1")
        row_axes[1].plot(sub["layer"], sub["image2_text_subspace_ratio"], label="image2")
        row_axes[1].set_title(f"{dataset}: Text-Subspace Projection")
        row_axes[1].set_xlabel("layer")
        row_axes[1].set_ylabel("projection ratio")
        row_axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cache_swap_step_attention(compare_df: pd.DataFrame, out_path: Path) -> None:
    if compare_df.empty:
        return
    grouped = compare_df.groupby(["dataset", "variant", "step"], as_index=False)[
        ["image1_mass", "image2_mass", "text_mass"]
    ].mean()
    datasets = grouped["dataset"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(datasets), 2, figsize=(12, 4 * len(datasets)), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for row_axes, dataset in zip(axes, datasets):
        dataset_df = grouped[grouped["dataset"] == dataset]
        for axis, variant in zip(row_axes, ["original", "swapped"]):
            sub = dataset_df[dataset_df["variant"] == variant]
            axis.plot(sub["step"], sub["image1_mass"], label="image1", color="#4c72b0")
            axis.plot(sub["step"], sub["image2_mass"], label="image2", color="#dd8452")
            axis.plot(sub["step"], sub["text_mass"], label="text", color="#55a868")
            axis.set_title(f"{dataset}: {variant}")
            axis.set_xlabel("decode step")
            axis.set_ylabel("attention mass")
            axis.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def relabel_cache_swap_by_content(compare_df: pd.DataFrame) -> pd.DataFrame:
    if compare_df.empty:
        return compare_df.copy()
    relabeled = compare_df.copy()
    swapped_mask = relabeled["variant"] == "swapped"
    image1_mass = relabeled.loc[swapped_mask, "image1_mass"].copy()
    image1_l2 = relabeled.loc[swapped_mask, "image1_l2"].copy()
    image1_max = relabeled.loc[swapped_mask, "image1_max"].copy()
    relabeled.loc[swapped_mask, "image1_mass"] = relabeled.loc[swapped_mask, "image2_mass"].to_numpy()
    relabeled.loc[swapped_mask, "image1_l2"] = relabeled.loc[swapped_mask, "image2_l2"].to_numpy()
    relabeled.loc[swapped_mask, "image1_max"] = relabeled.loc[swapped_mask, "image2_max"].to_numpy()
    relabeled.loc[swapped_mask, "image2_mass"] = image1_mass.to_numpy()
    relabeled.loc[swapped_mask, "image2_l2"] = image1_l2.to_numpy()
    relabeled.loc[swapped_mask, "image2_max"] = image1_max.to_numpy()
    return relabeled


def plot_cache_swap_step_attention_by_content(compare_df: pd.DataFrame, out_path: Path) -> None:
    relabeled = relabel_cache_swap_by_content(compare_df)
    if relabeled.empty:
        return
    grouped = relabeled.groupby(["dataset", "variant", "step"], as_index=False)[
        ["image1_mass", "image2_mass", "text_mass"]
    ].mean()
    datasets = grouped["dataset"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(datasets), 2, figsize=(12, 4 * len(datasets)), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for row_axes, dataset in zip(axes, datasets):
        dataset_df = grouped[grouped["dataset"] == dataset]
        for axis, variant in zip(row_axes, ["original", "swapped"]):
            sub = dataset_df[dataset_df["variant"] == variant]
            axis.plot(sub["step"], sub["image1_mass"], label="image1 content", color="#4c72b0")
            axis.plot(sub["step"], sub["image2_mass"], label="image2 content", color="#dd8452")
            axis.plot(sub["step"], sub["text_mass"], label="text", color="#55a868")
            axis.set_title(f"{dataset}: {variant} (content relabeled)")
            axis.set_xlabel("decode step")
            axis.set_ylabel("attention mass")
            axis.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cache_swap_step0_per_sample(compare_df: pd.DataFrame, out_path: Path) -> None:
    if compare_df.empty:
        return
    step0 = compare_df[compare_df["step"] == 0]
    if step0.empty:
        return
    grouped = (
        step0.groupby(["dataset", "sample_index", "variant"], as_index=False)[["image1_mass", "image2_mass"]]
        .mean()
    )
    datasets = grouped["dataset"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(datasets), 2, figsize=(12, 4 * len(datasets)), sharex=False, sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for row_axes, dataset in zip(axes, datasets):
        dataset_df = grouped[grouped["dataset"] == dataset]
        for axis, variant in zip(row_axes, ["original", "swapped"]):
            sub = dataset_df[dataset_df["variant"] == variant]
            axis.scatter(sub["image1_mass"], sub["image2_mass"], alpha=0.6, s=18, color="#4c72b0")
            lo = min(sub["image1_mass"].min(), sub["image2_mass"].min())
            hi = max(sub["image1_mass"].max(), sub["image2_mass"].max())
            axis.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.0)
            prefers = (
                int((sub["image2_mass"] > sub["image1_mass"]).sum())
                if variant == "original"
                else int((sub["image1_mass"] > sub["image2_mass"]).sum())
            )
            axis.set_title(f"{dataset}: {variant} (flip-count={prefers})")
            axis.set_xlabel("image1 mass")
            axis.set_ylabel("image2 mass")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cache_swap_step0_per_sample_by_content(compare_df: pd.DataFrame, out_path: Path) -> None:
    relabeled = relabel_cache_swap_by_content(compare_df)
    if relabeled.empty:
        return
    step0 = relabeled[relabeled["step"] == 0]
    if step0.empty:
        return
    grouped = (
        step0.groupby(["dataset", "sample_index", "variant"], as_index=False)[["image1_mass", "image2_mass"]]
        .mean()
    )
    datasets = grouped["dataset"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(datasets), 2, figsize=(12, 4 * len(datasets)), sharex=False, sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for row_axes, dataset in zip(axes, datasets):
        dataset_df = grouped[grouped["dataset"] == dataset]
        for axis, variant in zip(row_axes, ["original", "swapped"]):
            sub = dataset_df[dataset_df["variant"] == variant]
            axis.scatter(sub["image1_mass"], sub["image2_mass"], alpha=0.6, s=18, color="#4c72b0")
            lo = min(sub["image1_mass"].min(), sub["image2_mass"].min())
            hi = max(sub["image1_mass"].max(), sub["image2_mass"].max())
            axis.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.0)
            prefers = int((sub["image2_mass"] > sub["image1_mass"]).sum())
            axis.set_title(f"{dataset}: {variant} (content, image2>image1={prefers})")
            axis.set_xlabel("image1 content mass")
            axis.set_ylabel("image2 content mass")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    image2_summary_paths = sorted(input_root.glob("image2_*/gpu*/summary.json"))
    image2_prefill_paths = sorted(input_root.glob("image2_*/gpu*/prefill_attention.csv"))
    image2_decode_paths = sorted(input_root.glob("image2_*/gpu*/decode_attention.csv"))
    image2_alignment_paths = sorted(input_root.glob("image2_*/gpu*/layer_alignment.csv"))
    cache_swap_summary_paths = sorted(input_root.glob("cache_swap_*/gpu*/summary.json"))
    cache_swap_compare_paths = sorted(input_root.glob("cache_swap_*/gpu*/cache_swap_compare.csv"))
    cache_swap_sample_paths = sorted(input_root.glob("cache_swap_*/gpu*/sample_summary.csv"))

    image2_summary_rows = []
    for path in image2_summary_paths:
        row = load_json(path)
        row["job"] = path.parent.parent.name
        row["shard"] = path.parent.name
        image2_summary_rows.append(row)
    image2_summary_df = pd.DataFrame(image2_summary_rows)

    image2_prefill_df = concat_csvs_with_dataset(image2_prefill_paths, "image2_")
    image2_decode_df = concat_csvs_with_dataset(image2_decode_paths, "image2_")
    image2_alignment_df = concat_csvs_with_dataset(image2_alignment_paths, "image2_")
    cache_swap_compare_df = concat_csvs_with_dataset(cache_swap_compare_paths, "cache_swap_")
    cache_swap_sample_df = concat_csvs_with_dataset(cache_swap_sample_paths, "cache_swap_")

    image2_prefill_df.to_csv(output_root / "image2_prefill_attention_merged.csv", index=False)
    image2_decode_df.to_csv(output_root / "image2_decode_attention_merged.csv", index=False)
    image2_alignment_df.to_csv(output_root / "image2_layer_alignment_merged.csv", index=False)
    image2_summary_df.to_csv(output_root / "image2_summary_merged.csv", index=False)
    cache_swap_compare_df.to_csv(output_root / "cache_swap_compare_merged.csv", index=False)
    cache_swap_sample_df.to_csv(output_root / "cache_swap_sample_summary_merged.csv", index=False)

    plot_prefill(image2_prefill_df, output_root / "prefill_last_token_attention.png")
    plot_decode(image2_decode_df, output_root / "decode_attention_over_steps.png")
    plot_alignment(image2_alignment_df, output_root / "layer_alignment_over_layers.png")
    plot_cache_swap_step_attention(cache_swap_compare_df, output_root / "cache_swap_step_attention.png")
    plot_cache_swap_step0_per_sample(cache_swap_compare_df, output_root / "cache_swap_step0_per_sample.png")
    plot_cache_swap_step_attention_by_content(
        cache_swap_compare_df, output_root / "cache_swap_step_attention_by_content.png"
    )
    plot_cache_swap_step0_per_sample_by_content(
        cache_swap_compare_df, output_root / "cache_swap_step0_per_sample_by_content.png"
    )

    cache_swap_compared = (
        int(cache_swap_sample_df.get("used_for_compare", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        if not cache_swap_sample_df.empty
        else 0
    )
    cache_swap_skipped = (
        int((cache_swap_sample_df.get("skip_reason", pd.Series(dtype=str)).fillna("") != "").sum())
        if not cache_swap_sample_df.empty
        else 0
    )

    summary = {
        "input_root": str(input_root),
        "image2": {
            "summary_shard_count": int(len(image2_summary_df)),
            "processed_sample_count": int(image2_summary_df["processed_sample_count"].sum()) if not image2_summary_df.empty else 0,
            "target_sample_count": int(image2_summary_df["sample_count"].sum()) if not image2_summary_df.empty else 0,
            "datasets": (
                image2_summary_df.groupby("dataset")[["processed_sample_count", "sample_count"]]
                .sum()
                .astype(int)
                .to_dict(orient="index")
                if not image2_summary_df.empty
                else {}
            ),
        },
        "cache_swap_partial": {
            "compare_row_count": int(len(cache_swap_compare_df)),
            "sample_row_count": int(len(cache_swap_sample_df)),
            "compared_sample_count": cache_swap_compared,
            "skipped_or_failed_sample_rows": cache_swap_skipped,
            "datasets": (
                cache_swap_sample_df.groupby("dataset").size().to_dict()
                if not cache_swap_sample_df.empty
                else {}
            ),
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
