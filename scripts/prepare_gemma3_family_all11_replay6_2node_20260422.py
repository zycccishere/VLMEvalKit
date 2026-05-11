#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


MATRIX_CONFIG = Path("scripts/configs/matrix_gemma3_family_all11_replay6_2node_20260422.yaml")
MODEL_CONFIG = Path("scripts/configs/models.yaml")
OUT_DIR = Path("scripts/configs/task_manifests/gemma3_family_all11_replay6_2node_20260422")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def task_weight(model_cfg: dict[str, Any], gpus_per_node: int) -> float:
    runtime = model_cfg["runtime"]
    gpus_per_job = int(runtime["gpus_per_job"])
    if gpus_per_job <= 0:
        raise ValueError(f"invalid gpus_per_job={gpus_per_job}")
    workers_per_node = max(1, gpus_per_node // gpus_per_job)
    return float(runtime["estimated_dataset_cost"]) / workers_per_node


def build_rows(
    matrix: dict[str, Any],
    models_cfg: dict[str, Any],
    nodes: int,
    gpus_per_node: int,
) -> tuple[list[dict[str, str]], list[float]]:
    rows: list[dict[str, str]] = []
    loads = [0.0 for _ in range(nodes)]
    tasks: list[tuple[float, int, int, int, int, dict[str, str]]] = []

    models = list(matrix["models"])
    policies = list(matrix["policies"].keys())
    modes = list(matrix["replay_modes"])
    transforms = list(matrix.get("image_transforms", ["baseline"]))
    datasets = list(matrix["datasets"])

    task_index = 0
    for model_idx, model in enumerate(models):
        model_cfg = models_cfg["models"][model]
        weight = task_weight(model_cfg, gpus_per_node=gpus_per_node)
        gpus_per_job = int(model_cfg["runtime"]["gpus_per_job"])
        for policy in policies:
            for mode_idx, mode in enumerate(modes):
                for transform in transforms:
                    for dataset_idx, dataset in enumerate(datasets):
                        row = {
                            "model": model,
                            "policy": policy,
                            "mode": mode,
                            "transform": transform,
                            "dataset": dataset,
                        }
                        tasks.append((weight, gpus_per_job, model_idx, mode_idx, dataset_idx, row))
                        task_index += 1

    # Longest-processing-time assignment keeps both nodes balanced while still
    # mixing 27B TP=2 tasks with 12B/4B single-GPU tasks in each shard.
    for weight, gpus_per_job, model_idx, mode_idx, dataset_idx, row in sorted(
        tasks,
        key=lambda item: (-item[0], -item[1], item[2], item[3], item[4]),
    ):
        node = min(range(nodes), key=lambda idx: (loads[idx], idx))
        out = dict(row)
        out["node"] = str(node)
        rows.append(out)
        loads[node] += weight

    model_order = {model: idx for idx, model in enumerate(models)}
    mode_order = {mode: idx for idx, mode in enumerate(modes)}
    dataset_order = {dataset: idx for idx, dataset in enumerate(datasets)}
    rows.sort(
        key=lambda row: (
            int(row["node"]),
            model_order[row["model"]],
            mode_order[row["mode"]],
            dataset_order[row["dataset"]],
        )
    )
    return rows, loads


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "policy", "mode", "transform", "dataset", "node"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare 2-node manifests for Gemma 3 family all11 replay6 runs.")
    parser.add_argument("--matrix-config", type=Path, default=MATRIX_CONFIG)
    parser.add_argument("--model-config", type=Path, default=MODEL_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    args = parser.parse_args()

    matrix = load_yaml(args.matrix_config)
    models_cfg = load_yaml(args.model_config)
    rows, loads = build_rows(matrix, models_cfg, nodes=args.nodes, gpus_per_node=args.gpus_per_node)

    write_csv(args.out_dir / "all_tasks.csv", rows)
    summary = []
    for node in range(args.nodes):
        node_rows = [row for row in rows if row["node"] == str(node)]
        write_csv(args.out_dir / f"node{node}_tasks.csv", node_rows)
        model_counts: dict[str, int] = {}
        dataset_counts: dict[str, int] = {}
        mode_counts: dict[str, int] = {}
        for row in node_rows:
            model_counts[row["model"]] = model_counts.get(row["model"], 0) + 1
            dataset_counts[row["dataset"]] = dataset_counts.get(row["dataset"], 0) + 1
            mode_counts[row["mode"]] = mode_counts.get(row["mode"], 0) + 1
        summary.append(
            {
                "node": node,
                "tasks": len(node_rows),
                "load": round(loads[node], 4),
                "models": dict(sorted(model_counts.items())),
                "datasets": dict(sorted(dataset_counts.items())),
                "modes": dict(sorted(mode_counts.items())),
            }
        )

    payload = {
        "matrix_config": str(args.matrix_config),
        "model_config": str(args.model_config),
        "out_dir": str(args.out_dir),
        "nodes": args.nodes,
        "gpus_per_node": args.gpus_per_node,
        "task_count": len(rows),
        "nodes_summary": summary,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
