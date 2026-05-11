from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


@dataclass
class Span:
    name: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def parse_attention_layers(spec: str, layer_count: int) -> list[int]:
    raw = (spec or "last").strip().lower()
    if raw == "last":
        return [layer_count - 1]
    if raw == "all":
        return list(range(layer_count))
    if raw.startswith("last") and raw[4:].isdigit():
        width = max(1, min(int(raw[4:]), layer_count))
        return list(range(layer_count - width, layer_count))

    layers: list[int] = []
    seen: set[int] = set()
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        idx = int(part)
        if idx < 0:
            idx += layer_count
        if idx < 0 or idx >= layer_count:
            raise ValueError(f"Layer index out of range for spec={spec!r}: {part}")
        if idx not in seen:
            layers.append(idx)
            seen.add(idx)
    if not layers:
        raise ValueError(f"Empty attention-layer selection: {spec!r}")
    return layers


def aggregate_attention(
    attn_last: torch.Tensor,
    spans: dict[str, list[int]],
    *,
    head_reduction: str = "per_head",
) -> list[dict[str, float]]:
    attn_last = attn_last.squeeze(0).squeeze(1).float()
    rows = []
    for head_idx, head_attn in enumerate(attn_last):
        row = {"head": int(head_idx)}
        for name, positions in spans.items():
            if not positions:
                row[f"{name}_mass"] = float("nan")
                row[f"{name}_l2"] = float("nan")
                row[f"{name}_max"] = float("nan")
                continue
            values = head_attn[positions]
            row[f"{name}_mass"] = float(values.sum().item())
            row[f"{name}_l2"] = float(torch.linalg.vector_norm(values, ord=2).item())
            row[f"{name}_max"] = float(values.max().item())
        rows.append(row)
    if head_reduction == "per_head" or not rows:
        return rows
    reduced = {"head": "mean"}
    for key in rows[0]:
        if key == "head":
            continue
        values = [row[key] for row in rows if row[key] == row[key]]
        reduced[key] = float(sum(values) / len(values)) if values else float("nan")
    return [reduced]


def append_attention_rows(
    *,
    sink: list[dict],
    attn_last: torch.Tensor,
    spans: dict[str, list[int]],
    sample_index: int,
    stage: str,
    layer: int,
    step: int,
    head_reduction: str,
    token_id: int | None = None,
    token_text: str | None = None,
) -> None:
    for head_row in aggregate_attention(attn_last, spans, head_reduction=head_reduction):
        head_row.update(
            {
                "sample_index": sample_index,
                "stage": stage,
                "layer": int(layer),
                "step": int(step),
                "token_id": token_id,
                "token_text": token_text,
            }
        )
        sink.append(head_row)


def filter_attention_layer(df: pd.DataFrame, layer: int) -> pd.DataFrame:
    if df.empty or "layer" not in df.columns:
        return df
    return df[df["layer"] == layer].copy()


def plot_prefill(prefill_df: pd.DataFrame, out_path: Path, *, summary_layer: int) -> None:
    if prefill_df.empty:
        return
    layer_df = filter_attention_layer(prefill_df, summary_layer)
    if layer_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    mass_cols = ["image1_mass", "image2_mass", "text_mass"]
    l2_cols = ["image1_l2", "image2_l2", "text_l2"]
    for ax, cols, title in zip(axes, [mass_cols, l2_cols], ["Prefill Mass", "Prefill L2 Norm"]):
        means = [layer_df[col].mean() for col in cols]
        stds = [layer_df[col].std(ddof=0) for col in cols]
        labels = [col.replace("_mass", "").replace("_l2", "") for col in cols]
        ax.bar(labels, means, yerr=stds, color=["#4878cf", "#ee854a", "#6acc64"])
        ax.set_title(f"{title} (layer {summary_layer})")
        ax.set_ylabel("attention")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_decode(decode_df: pd.DataFrame, out_path: Path, *, summary_layer: int) -> None:
    if decode_df.empty:
        return
    layer_df = filter_attention_layer(decode_df, summary_layer)
    if layer_df.empty:
        return
    grouped = layer_df.groupby("step", as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(grouped["step"], grouped["image1_mass"], label="image1")
    axes[0].plot(grouped["step"], grouped["image2_mass"], label="image2")
    axes[0].plot(grouped["step"], grouped["text_mass"], label="text")
    axes[0].set_title(f"Decode Attention Mass (layer {summary_layer})")
    axes[0].set_xlabel("decode step")
    axes[0].set_ylabel("mean head/sample mass")
    axes[0].legend()

    ratio = grouped["image2_mass"] / grouped["image1_mass"].clip(lower=1e-8)
    axes[1].plot(grouped["step"], ratio, color="#ee854a")
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    axes[1].set_title(f"Decode image2/image1 Mass Ratio (layer {summary_layer})")
    axes[1].set_xlabel("decode step")
    axes[1].set_ylabel("ratio")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_decode_ratio_by_layer(decode_df: pd.DataFrame, out_path: Path) -> None:
    if decode_df.empty or "layer" not in decode_df.columns:
        return
    grouped = decode_df.groupby(["layer", "step"], as_index=False).mean(numeric_only=True)
    if grouped.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for layer in sorted(grouped["layer"].unique()):
        layer_df = grouped[grouped["layer"] == layer]
        ratio = layer_df["image2_mass"] / layer_df["image1_mass"].clip(lower=1e-8)
        ax.plot(layer_df["step"], ratio, label=f"L{layer}")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_title("Decode image2/image1 Mass Ratio By Layer")
    ax.set_xlabel("decode step")
    ax.set_ylabel("ratio")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
