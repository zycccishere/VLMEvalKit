#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


MODEL_KEY = "qwen25vl_32b"
POLICY = "default"
REGISTRY_NAME = "Qwen2VLChatReplay"
UNIFIED_RESULTS_ROOT = Path("runs/by_setting_qwen25vl32b_unified_topology_image2_20260404")
OLD_TRANSFORM_ROOT = Path("runs/by_setting_qwen25vl32b_image2_transforms_20260401")
OLD_IMAGE_TEXT_ROOT = Path("runs/by_setting_qwen25vl32b_image_text_baseline_20260403")
MANIFEST_DIR = Path("scripts/configs/task_manifests/qwen25vl32b_unified_topology_image2_plus_20260404")
LAUNCH_DIR = Path("tmp/qwen25vl32b_unified_topology_image2_plus_20260404")

DATASET_COUNTS = {
    "DynaMath": 5010,
    "MathVision": 3040,
    "LogicVista": 447,
    "VisualPuzzles": 1168,
    "AI2D_TEST": 3088,
    "OCRBench": 1000,
    "SEEDBench2_Plus": 2277,
    "MathVista_MINI": 1000,
    "VisuLogic": 1000,
    "MMMU_DEV_VAL": 1050,
}

MODE_FACTORS = {
    "image_text": 1.0,
    "text_image": 1.0,
    "image_image_text": 2.0,
    "image_text_image": 2.0,
    "image_text_text": 1.1,
    "image_text_image_text": 2.1,
}

BASELINE_DATASETS = [
    "DynaMath",
    "MathVision",
    "LogicVista",
    "VisualPuzzles",
    "AI2D_TEST",
    "OCRBench",
    "SEEDBench2_Plus",
    "MathVista_MINI",
    "VisuLogic",
    "MMMU_DEV_VAL",
]

TRANSFORM_DATASETS = [
    "DynaMath",
    "MathVision",
    "LogicVista",
    "VisualPuzzles",
    "AI2D_TEST",
    "OCRBench",
    "SEEDBench2_Plus",
    "MathVista_MINI",
    "VisuLogic",
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
    kind: str
    mode: str
    transform: str
    dataset: str

    @property
    def weight(self) -> float:
        return DATASET_COUNTS[self.dataset] * MODE_FACTORS[self.mode]

    def as_csv_row(self, status: str) -> dict[str, str]:
        return {
            "model_key": MODEL_KEY,
            "policy": POLICY,
            "mode": self.mode,
            "transform": self.transform,
            "dataset": self.dataset,
            "kind": self.kind,
            "status": status,
            "weight": f"{self.weight:.2f}",
        }


def metric_exists(task_root: Path, dataset: str) -> bool:
    model_output = task_root / REGISTRY_NAME
    for pattern in (f"*{dataset}*_acc.csv", f"*{dataset}*_score.csv", f"*{dataset}*_score.json"):
        if any(model_output.glob(pattern)):
            return True
    return False


def result_exists(task_root: Path, dataset: str) -> bool:
    model_output = task_root / REGISTRY_NAME
    pred_file = model_output / f"{REGISTRY_NAME}_{dataset}.xlsx"
    return pred_file.exists() and metric_exists(task_root, dataset)


def unified_task_root(task: TaskRow) -> Path:
    return UNIFIED_RESULTS_ROOT / POLICY / task.mode / task.transform / MODEL_KEY


def source_task_root(task: TaskRow) -> Path | None:
    if task.kind == "replace":
        return OLD_TRANSFORM_ROOT / POLICY / "image_text_image" / task.transform / MODEL_KEY
    if task.mode == "image_text":
        return OLD_IMAGE_TEXT_ROOT / POLICY / "image_text" / MODEL_KEY
    if task.mode == "image_text_image":
        return OLD_TRANSFORM_ROOT / POLICY / "image_text_image" / "baseline" / MODEL_KEY
    return None


def desired_tasks() -> list[TaskRow]:
    rows: list[TaskRow] = []
    for mode in BASELINE_MODES:
        for dataset in BASELINE_DATASETS:
            rows.append(TaskRow(kind="baseline", mode=mode, transform="baseline", dataset=dataset))
    for transform in REPLACE_TRANSFORMS:
        for dataset in TRANSFORM_DATASETS:
            rows.append(TaskRow(kind="replace", mode="image_text_image", transform=transform, dataset=dataset))
    return rows


def classify_task(task: TaskRow) -> str:
    if result_exists(unified_task_root(task), task.dataset):
        return "reuse_full"
    source = source_task_root(task)
    if source is not None and result_exists(source, task.dataset):
        return "reuse_full"
    return "missing"


def unique_copy_mappings(tasks: list[TaskRow]) -> list[tuple[Path, Path]]:
    seen: set[tuple[Path, Path]] = set()
    out: list[tuple[Path, Path]] = []
    for task in tasks:
        if classify_task(task) != "reuse_full":
            continue
        src = source_task_root(task)
        dst = unified_task_root(task)
        if src is None or not src.exists() or src == dst:
            continue
        pair = (src, dst)
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def sync_reused_results(tasks: list[TaskRow]) -> list[dict[str, str]]:
    ops: list[dict[str, str]] = []
    for src, dst in unique_copy_mappings(tasks):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        ops.append({"src": str(src), "dst": str(dst)})
    return ops


def greedy_split(tasks: list[TaskRow], nodes: int) -> tuple[list[list[TaskRow]], list[float]]:
    ordered = sorted(tasks, key=lambda item: (-item.weight, item.mode, item.transform, item.dataset))
    buckets: list[list[TaskRow]] = [[] for _ in range(nodes)]
    loads = [0.0 for _ in range(nodes)]
    for task in ordered:
        node_idx = min(range(nodes), key=lambda idx: (loads[idx], idx))
        buckets[node_idx].append(task)
        loads[node_idx] += task.weight
    for bucket in buckets:
        bucket.sort(key=lambda item: (item.kind, item.mode, item.transform, item.dataset))
    return buckets, loads


def write_csv(path: Path, tasks: list[TaskRow], status_lookup: dict[TaskRow, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model_key", "policy", "mode", "transform", "dataset", "kind", "status", "weight"],
        )
        writer.writeheader()
        for task in tasks:
            writer.writerow(task.as_csv_row(status_lookup[task]))


def write_launch_script(path: Path, repo_root: Path, matrix_config: Path, manifest_path: Path) -> None:
    text = f"""#!/usr/bin/env bash
set -euo pipefail

cd "{repo_root}"
NODE_RANK=0 NUM_NODES=1 REPLAY_TRACE_LEVEL=summary REPLAY_TRACE_SAMPLES=1 \\
bash scripts/run_benchmark.sh \\
  --matrix-config {matrix_config.as_posix()} \\
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
    parser = argparse.ArgumentParser(description="Prepare extended unified Qwen2.5-VL topology/image2 result root and 4-node manifests.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--sync-results", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    tasks = desired_tasks()
    status_lookup = {task: classify_task(task) for task in tasks}
    missing_tasks = [task for task in tasks if status_lookup[task] == "missing"]
    reuse_tasks = [task for task in tasks if status_lookup[task] == "reuse_full"]
    sync_ops = sync_reused_results(tasks) if args.sync_results else []
    if args.sync_results:
        status_lookup = {task: classify_task(task) for task in tasks}
        missing_tasks = [task for task in tasks if status_lookup[task] == "missing"]
        reuse_tasks = [task for task in tasks if status_lookup[task] == "reuse_full"]

    buckets, loads = greedy_split(missing_tasks, args.nodes)

    manifest_dir = repo_root / MANIFEST_DIR
    launch_dir = repo_root / LAUNCH_DIR
    matrix_config = Path("scripts/configs/matrix_qwen25vl32b_unified_topology_image2_plus_20260404.yaml")

    write_csv(manifest_dir / "all_tasks.csv", tasks, status_lookup)
    write_csv(manifest_dir / "reuse_tasks.csv", reuse_tasks, status_lookup)
    write_csv(manifest_dir / "missing_tasks.csv", missing_tasks, status_lookup)
    for node_idx, bucket in enumerate(buckets):
        manifest_path = manifest_dir / f"node{node_idx}_tasks.csv"
        write_csv(manifest_path, bucket, status_lookup)
        write_launch_script(launch_dir / f"run_node{node_idx}.sh", repo_root, matrix_config, manifest_path.relative_to(repo_root))

    summary = {
        "repo_root": str(repo_root),
        "results_root": str((repo_root / UNIFIED_RESULTS_ROOT).resolve()),
        "matrix_config": str((repo_root / matrix_config).resolve()),
        "task_counts": {
            "all": len(tasks),
            "reuse_full": len(reuse_tasks),
            "missing": len(missing_tasks),
        },
        "copy_mappings": sync_ops,
        "nodes": [
            {
                "node": idx,
                "task_count": len(bucket),
                "estimated_load": round(loads[idx], 2),
                "manifest": str((manifest_dir / f"node{idx}_tasks.csv").resolve()),
                "launch_script": str((launch_dir / f"run_node{idx}.sh").resolve()),
            }
            for idx, bucket in enumerate(buckets)
        ],
    }
    summary_path = manifest_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
