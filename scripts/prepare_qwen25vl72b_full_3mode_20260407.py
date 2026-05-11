#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


MODEL_KEY = "qwen25vl_72b"
MODEL_DISPLAY = "Qwen2.5-VL-72B"
POLICY = "default"
MATRIX_CONFIG = Path("scripts/configs/matrix_qwen25vl72b_full_3mode_20260407.yaml")
RESULTS_ROOT = Path("runs/by_setting_qwen25vl72b_full_3mode_20260407")
MANIFEST_DIR = Path("scripts/configs/task_manifests/qwen25vl72b_full_3mode_20260407")
LAUNCH_DIR = Path("tmp/qwen25vl72b_full_3mode_20260407")

DATASET_COUNTS = {
    "AI2D_TEST": 3088,
    "MathVista_MINI": 1000,
    "OCRBench": 1000,
    "SEEDBench2_Plus": 2277,
    "VisuLogic": 1000,
    "LogicVista": 447,
    "VisualPuzzles": 1168,
    "DynaMath": 5010,
    "MathVision": 3040,
}

MODE_FACTORS = {
    "image_text": 1.0,
    "image_text_image": 2.0,
    "image_image_text": 2.0,
}

DATASETS = [
    "AI2D_TEST",
    "MathVista_MINI",
    "OCRBench",
    "SEEDBench2_Plus",
    "VisuLogic",
    "LogicVista",
    "VisualPuzzles",
    "DynaMath",
    "MathVision",
]

MODES = [
    "image_text",
    "image_text_image",
    "image_image_text",
]


@dataclass(frozen=True)
class TaskRow:
    mode: str
    dataset: str
    transform: str = "baseline"

    @property
    def weight(self) -> float:
        return DATASET_COUNTS[self.dataset] * MODE_FACTORS[self.mode]

    def as_csv_row(self) -> dict[str, str]:
        return {
            "model_key": MODEL_KEY,
            "policy": POLICY,
            "mode": self.mode,
            "transform": self.transform,
            "dataset": self.dataset,
            "kind": "baseline",
            "status": "missing",
            "weight": f"{self.weight:.2f}",
        }


def desired_tasks() -> list[TaskRow]:
    rows: list[TaskRow] = []
    for mode in MODES:
        for dataset in DATASETS:
            rows.append(TaskRow(mode=mode, dataset=dataset))
    return rows


def greedy_split(tasks: list[TaskRow], nodes: int) -> tuple[list[list[TaskRow]], list[float]]:
    ordered = sorted(tasks, key=lambda item: (-item.weight, item.mode, item.dataset))
    buckets: list[list[TaskRow]] = [[] for _ in range(nodes)]
    loads = [0.0 for _ in range(nodes)]
    for task in ordered:
        node_idx = min(range(nodes), key=lambda idx: (loads[idx], idx))
        buckets[node_idx].append(task)
        loads[node_idx] += task.weight
    for bucket in buckets:
        bucket.sort(key=lambda item: (item.mode, item.dataset))
    return buckets, loads


def write_csv(path: Path, tasks: list[TaskRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model_key", "policy", "mode", "transform", "dataset", "kind", "status", "weight"],
        )
        writer.writeheader()
        for task in tasks:
            writer.writerow(task.as_csv_row())


def write_launch_script(path: Path, repo_root: Path, manifest_path: Path) -> None:
    text = f"""#!/usr/bin/env bash
set -euo pipefail

cd "{repo_root}"
NODE_RANK=0 NUM_NODES=1 REPLAY_TRACE_LEVEL=summary REPLAY_TRACE_SAMPLES=1 \\
bash scripts/run_benchmark.sh \\
  --matrix-config {MATRIX_CONFIG.as_posix()} \\
  --task-manifest {manifest_path.as_posix()} \\
  --nodes 1 \\
  --node-rank 0 \\
  --gpu-ids 0,1,2,3,4,5,6,7 \\
  --resume-infer
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare 3-node manifests for qwen25vl_72b full 3-mode baseline runs.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--nodes", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    tasks = desired_tasks()
    buckets, loads = greedy_split(tasks, args.nodes)

    manifest_dir = repo_root / MANIFEST_DIR
    launch_dir = repo_root / LAUNCH_DIR

    write_csv(manifest_dir / "all_tasks.csv", tasks)

    summary_nodes = []
    for node_idx, bucket in enumerate(buckets):
        manifest_path = manifest_dir / f"node{node_idx}_tasks.csv"
        write_csv(manifest_path, bucket)
        write_launch_script(launch_dir / f"run_node{node_idx}.sh", repo_root, manifest_path.relative_to(repo_root))
        summary_nodes.append(
            {
                "node": node_idx,
                "model_key": MODEL_KEY,
                "model_display": MODEL_DISPLAY,
                "task_count": len(bucket),
                "estimated_load": round(loads[node_idx], 2),
                "manifest": str(manifest_path.resolve()),
                "launch_script": str((launch_dir / f"run_node{node_idx}.sh").resolve()),
            }
        )

    summary = {
        "repo_root": str(repo_root),
        "results_root": str((repo_root / RESULTS_ROOT).resolve()),
        "matrix_config": str((repo_root / MATRIX_CONFIG).resolve()),
        "task_count": len(tasks),
        "nodes": summary_nodes,
    }
    summary_path = manifest_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
