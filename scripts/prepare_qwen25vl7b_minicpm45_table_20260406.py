#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


POLICY = "default"
MATRIX_CONFIG = Path("scripts/configs/matrix_qwen25vl7b_minicpm45_table_20260406.yaml")
RESULTS_ROOT = Path("runs/by_setting_qwen25vl7b_minicpm45_table_20260406")
MANIFEST_DIR = Path("scripts/configs/task_manifests/qwen25vl7b_minicpm45_table_20260406")
LAUNCH_DIR = Path("tmp/qwen25vl7b_minicpm45_table_20260406")

MODEL_TO_NODE = {
    "qwen25vl_7b": 0,
    "minicpm_v_45": 1,
    "minicpm_o_45": 2,
}

MODEL_DISPLAY = {
    "qwen25vl_7b": "Qwen2.5-VL-7B",
    "minicpm_v_45": "MiniCPM-V-4_5",
    "minicpm_o_45": "MiniCPM-o-4_5",
}

DATASET_COUNTS = {
    "MathVision": 3040,
    "DynaMath": 5010,
    "LogicVista": 447,
    "VisualPuzzles": 1168,
    "MathVista_MINI": 1000,
    "AI2D_TEST": 3088,
    "OCRBench": 1000,
    "SEEDBench2_Plus": 2277,
}

MODE_FACTORS = {
    "image_text": 1.0,
    "text_image": 1.0,
    "image_image_text": 2.0,
    "image_text_image": 2.0,
    "image_text_text": 1.1,
    "image_text_image_text": 2.1,
}

DATASETS = [
    "MathVision",
    "DynaMath",
    "LogicVista",
    "VisualPuzzles",
    "MathVista_MINI",
    "AI2D_TEST",
    "OCRBench",
    "SEEDBench2_Plus",
]

BASELINE_MODES = [
    "image_text",
    "text_image",
    "image_image_text",
    "image_text_image",
    "image_text_text",
    "image_text_image_text",
]

REPLACE_TRANSFORMS = [
    "mask10_white",
    "mask20_white",
    "blank",
    "rotate180",
    "shift_right_halfpatch_wrap",
    "shift_right_onepatch_wrap",
    "shift_down_halfpatch_wrap",
    "shift_down_onepatch_wrap",
    "zoom_1p5_uncropped",
]


@dataclass(frozen=True)
class TaskRow:
    model_key: str
    kind: str
    mode: str
    transform: str
    dataset: str

    @property
    def weight(self) -> float:
        return DATASET_COUNTS[self.dataset] * MODE_FACTORS[self.mode]

    def as_csv_row(self) -> dict[str, str]:
        return {
            "model_key": self.model_key,
            "policy": POLICY,
            "mode": self.mode,
            "transform": self.transform,
            "dataset": self.dataset,
            "kind": self.kind,
            "status": "missing",
            "weight": f"{self.weight:.2f}",
        }


def desired_tasks() -> list[TaskRow]:
    rows: list[TaskRow] = []
    for model_key in MODEL_TO_NODE:
        for mode in BASELINE_MODES:
            for dataset in DATASETS:
                rows.append(TaskRow(model_key=model_key, kind="baseline", mode=mode, transform="baseline", dataset=dataset))
        for transform in REPLACE_TRANSFORMS:
            for dataset in DATASETS:
                rows.append(TaskRow(model_key=model_key, kind="replace", mode="image_text_image", transform=transform, dataset=dataset))
    return rows


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


def write_launch_script(path: Path, repo_root: Path, manifest_path: Path, model_key: str) -> None:
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
    parser = argparse.ArgumentParser(description="Prepare 3-node manifests for qwen25vl_7b + MiniCPM-4.5 table runs.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    tasks = desired_tasks()
    buckets: dict[int, list[TaskRow]] = {idx: [] for idx in sorted(set(MODEL_TO_NODE.values()))}
    for task in tasks:
        buckets[MODEL_TO_NODE[task.model_key]].append(task)

    manifest_dir = repo_root / MANIFEST_DIR
    launch_dir = repo_root / LAUNCH_DIR

    write_csv(manifest_dir / "all_tasks.csv", tasks)
    summary_nodes = []
    for model_key, node_idx in MODEL_TO_NODE.items():
        node_tasks = buckets[node_idx]
        node_tasks = [task for task in node_tasks if task.model_key == model_key]
        manifest_path = manifest_dir / f"node{node_idx}_tasks.csv"
        write_csv(manifest_path, node_tasks)
        write_launch_script(
            launch_dir / f"run_node{node_idx}.sh",
            repo_root,
            manifest_path.relative_to(repo_root),
            model_key,
        )
        summary_nodes.append(
            {
                "node": node_idx,
                "model_key": model_key,
                "model_display": MODEL_DISPLAY[model_key],
                "task_count": len(node_tasks),
                "estimated_load": round(sum(task.weight for task in node_tasks), 2),
                "manifest": str((manifest_dir / f"node{node_idx}_tasks.csv").resolve()),
                "launch_script": str((launch_dir / f"run_node{node_idx}.sh").resolve()),
            }
        )

    summary = {
        "repo_root": str(repo_root),
        "results_root": str((repo_root / RESULTS_ROOT).resolve()),
        "matrix_config": str((repo_root / MATRIX_CONFIG).resolve()),
        "task_count": len(tasks),
        "nodes": sorted(summary_nodes, key=lambda x: x["node"]),
    }
    summary_path = manifest_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
