#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


DEFAULT_VARIANTS = {
    "off": {
        "gemma4_e2b_8192",
        "gemma4_e4b_8192",
        "gemma4_26b_a4b_8192",
        "gemma4_31b_8192",
    },
    "on": {
        "gemma4_e2b_think_on_nb8192",
        "gemma4_e4b_think_on_nb8192",
        "gemma4_26b_a4b_think_on_nb8192",
        "gemma4_31b_think_on_nb8192",
    },
}
DEFAULT_VARIANTS["both"] = DEFAULT_VARIANTS["off"] | DEFAULT_VARIANTS["on"]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping yaml: {path}")
    return data


def split_names(raw: str) -> list[str]:
    if not raw:
        return []
    return [part for part in raw.replace(",", " ").split() if part]


def task_weight(model_cfg: dict[str, Any], gpus_per_node: int) -> float:
    runtime = model_cfg["runtime"]
    gpus_per_job = int(runtime["gpus_per_job"])
    if gpus_per_job <= 0:
        raise ValueError(f"invalid gpus_per_job={gpus_per_job}")
    workers = max(1, gpus_per_node // gpus_per_job)
    return float(runtime["estimated_dataset_cost"]) / workers


def choose_chunk_count(group_load: float, target_load: float, task_count: int) -> int:
    if task_count <= 1:
        return 1
    if target_load <= 0:
        return 1
    split_threshold = max(target_load * 0.6, 1e-9)
    chunks = max(1, math.ceil(group_load / split_threshold))
    return min(task_count, chunks)


def strided_chunks(rows: list[dict[str, str]], chunk_count: int) -> list[list[dict[str, str]]]:
    if chunk_count <= 1:
        return [rows]
    chunks: list[list[dict[str, str]]] = [[] for _ in range(chunk_count)]
    for idx, row in enumerate(rows):
        chunks[idx % chunk_count].append(row)
    return [chunk for chunk in chunks if chunk]


def manifest_root(script_dir: Path, stem: str, variant: str) -> Path:
    return script_dir / "configs" / "task_manifests" / stem / variant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare balanced 4-node manifests for Gemma4 replay full runs.")
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--matrix-config",
        type=Path,
        default=script_dir / "configs" / "matrix_gemma4_identity_replay6_active9_toggle_nb8192_20260419.yaml",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=script_dir / "configs" / "models.yaml",
    )
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--variant", choices=["off", "on", "both"], default="both")
    parser.add_argument("--models", type=str, default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    matrix = load_yaml(args.matrix_config)
    model_cfg = load_yaml(args.model_config)

    explicit_models = set(split_names(args.models))
    allowed_models = explicit_models or DEFAULT_VARIANTS[args.variant]
    selected_models = [name for name in matrix["models"] if name in allowed_models]
    if not selected_models:
        raise SystemExit("no models selected for manifest generation")

    datasets = list(matrix["datasets"])
    policies = list(matrix["policies"].keys())
    replay_modes = list(matrix["replay_modes"])
    transforms = list(matrix.get("image_transforms", ["baseline"]))

    all_rows: list[dict[str, str]] = []
    per_model_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    per_model_weight: dict[str, float] = {}
    total_load = 0.0
    for model_name in selected_models:
        spec = model_cfg["models"][model_name]
        weight = task_weight(spec, gpus_per_node=args.gpus_per_node)
        per_model_weight[model_name] = weight
        for policy in policies:
            for mode in replay_modes:
                for transform in transforms:
                    for dataset in datasets:
                        row = {
                            "model": model_name,
                            "policy": policy,
                            "mode": mode,
                            "transform": transform,
                            "dataset": dataset,
                        }
                        per_model_rows[model_name].append(row)
                        all_rows.append(row)
        total_load += len(per_model_rows[model_name]) * weight

    target_load = total_load / max(1, args.nodes)
    groups: list[tuple[str, list[dict[str, str]], float]] = []
    for model_name in selected_models:
        rows = per_model_rows[model_name]
        load = len(rows) * per_model_weight[model_name]
        chunk_count = choose_chunk_count(load, target_load, len(rows))
        for chunk in strided_chunks(rows, chunk_count):
            groups.append(
                (
                    model_name,
                    chunk,
                    len(chunk) * per_model_weight[model_name],
                )
            )

    buckets: list[list[dict[str, str]]] = [[] for _ in range(args.nodes)]
    loads = [0.0 for _ in range(args.nodes)]
    for model_name, chunk, load in sorted(groups, key=lambda item: (-item[2], item[0], item[1][0]["mode"], item[1][0]["dataset"])):
        node_idx = min(range(args.nodes), key=lambda idx: (loads[idx], idx))
        buckets[node_idx].extend(chunk)
        loads[node_idx] += load

    script_dir = Path(__file__).resolve().parent
    matrix_stem = args.matrix_config.stem
    out_root = manifest_root(script_dir, matrix_stem, args.variant)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for node_idx, bucket in enumerate(buckets):
        path = out_root / f"node{node_idx}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in bucket:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        model_counts = Counter(row["model"] for row in bucket)
        dataset_counts = Counter(row["dataset"] for row in bucket)
        summary_rows.append(
            {
                "node": node_idx,
                "tasks": len(bucket),
                "load": round(loads[node_idx], 4),
                "models": dict(sorted(model_counts.items())),
                "datasets": dict(sorted(dataset_counts.items())),
                "manifest": str(path),
            }
        )

    summary = {
        "matrix": str(args.matrix_config),
        "variant": args.variant,
        "nodes": args.nodes,
        "gpus_per_node": args.gpus_per_node,
        "model_count": len(selected_models),
        "task_count": len(all_rows),
        "target_load": round(target_load, 4),
        "selected_models": selected_models,
        "nodes_summary": summary_rows,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
