#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


DEFAULT_MODEL_KEYS = [
    "qwen35_4b",
    "qwen35_9b",
    "qwen35_27b",
    "qwen35_35b_a3b",
]

DEFAULT_REASONING_DATASETS = [
    "MathVista_MINI",
    "VisuLogic",
    "LogicVista",
    "VisualPuzzles",
    "DynaMath",
    "MathVision",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backup and purge current Qwen3.5 default-policy reasoning results from runs/by_setting.")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, default=None)
    parser.add_argument("--policy", type=str, default="default")
    parser.add_argument("--model-keys", type=str, default=",".join(DEFAULT_MODEL_KEYS))
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_REASONING_DATASETS))
    return parser.parse_args()


def split_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_root = args.backup_root
    if backup_root is None:
        backup_root = args.runs_root.parent / "backups" / f"qwen35_default_reasoning_before_nothink_{timestamp}"
    runs_root = args.runs_root.resolve()
    backup_root = backup_root.resolve()
    model_keys = set(split_csv(args.model_keys))
    datasets = split_csv(args.datasets)

    moved_files: list[dict[str, str]] = []
    skipped_missing = 0

    policy_root = runs_root / args.policy
    for setting_dir in sorted(path for path in policy_root.iterdir() if path.is_dir()):
        for model_dir in sorted(path for path in setting_dir.iterdir() if path.is_dir()):
            if model_dir.name not in model_keys:
                continue
            for registry_dir in sorted(path for path in model_dir.iterdir() if path.is_dir() and not path.name.startswith("_")):
                registry_name = registry_dir.name
                for dataset in datasets:
                    matches = sorted(registry_dir.glob(f"{registry_name}_{dataset}*"))
                    if not matches:
                        skipped_missing += 1
                        continue
                    for path in matches:
                        rel = path.relative_to(runs_root)
                        target = backup_root / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(path), str(target))
                        moved_files.append(
                            {
                                "dataset": dataset,
                                "setting": setting_dir.name,
                                "model_key": model_dir.name,
                                "registry_name": registry_name,
                                "source": str(path),
                                "backup": str(target),
                            }
                        )

    payload = {
        "runs_root": str(runs_root),
        "backup_root": str(backup_root),
        "policy": args.policy,
        "model_keys": sorted(model_keys),
        "datasets": datasets,
        "moved_file_count": len(moved_files),
        "skipped_missing_count": skipped_missing,
        "moved_files": moved_files,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
