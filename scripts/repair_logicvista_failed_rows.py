#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair LogicVista rows that failed judge parsing.")
    parser.add_argument("--matrix-config", required=True)
    parser.add_argument("--model-config", default="scripts/configs/models.yaml")
    parser.add_argument("--judge", default=None)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = load_yaml(Path(args.matrix_config))
    model_cfg = load_yaml(Path(args.model_config))
    repo_root = Path(matrix["repo_root"])
    results_root_raw = Path(matrix["results_root"])
    results_root = results_root_raw if results_root_raw.is_absolute() else repo_root / results_root_raw

    eval_cfg = matrix.get("evaluation", {}) or {}
    model_name = str(args.judge or eval_cfg.get("judge", "gpt-4o-mini"))
    api_key = (
        os.environ.get("OPENAI_API_KEY_JUDGE", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("OPENAI_COMPATIBLE_API_KEY", "").strip()
        or str(eval_cfg.get("openai_api_key", "")).strip()
    )
    api_base = (
        os.environ.get("OPENAI_API_BASE_JUDGE", "").strip()
        or os.environ.get("OPENAI_API_BASE", "").strip()
        or os.environ.get("OPENAI_COMPATIBLE_API_BASE", "").strip()
        or str(eval_cfg.get("openai_api_base", "")).strip()
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_KEY_JUDGE"] = api_key
    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base
        os.environ["OPENAI_API_BASE_JUDGE"] = api_base
    os.environ["PYTHONPATH"] = str(repo_root) + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")

    import sys
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from vlmeval.dataset.utils.judge_util import build_judge
    from vlmeval.dataset.utils.logicvista import LogicVista_auxeval, evaluate_logicvista
    from vlmeval.smp import dump

    judge = build_judge(model=model_name)
    files = sorted(results_root.glob("**/Qwen2VLChatReplay/Qwen2VLChatReplay_LogicVista_gpt4o-mini.xlsx"))
    if args.limit_files is not None:
        files = files[: args.limit_files]

    total_repairs = 0
    for path in files:
        df = pd.read_excel(path)
        if "log" not in df.columns:
            continue
        bad_mask = df["log"].fillna("").str.contains("All 5 retries failed", na=False)
        bad = df[bad_mask].copy()
        if bad.empty:
            continue

        rel = path.relative_to(results_root)
        print(f"[FILE] {rel} bad_rows={len(bad)}", flush=True)
        if args.dry_run:
            continue

        backup = path.with_suffix(path.suffix + f".repairbak_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        if not backup.exists():
            shutil.copy2(path, backup)

        for row_idx, row in bad.iterrows():
            result = LogicVista_auxeval(judge, row)
            df.at[row_idx, "log"] = result["log"]
            df.at[row_idx, "res"] = result["res"]
            df.at[row_idx, "hit"] = result["hit"]
            print(
                f"  [ROW] index={row['index']} answer={row['answer']} new_res={result['res']} hit={result['hit']}",
                flush=True,
            )
            total_repairs += 1

        dump(df, str(path))
        score = evaluate_logicvista(str(path))
        score_path = path.with_name(path.stem + "_score.csv")
        score.to_csv(score_path, index=False)
        print(f"  [UPDATED] {score_path.relative_to(results_root)}", flush=True)

    print(f"[DONE] repaired_rows={total_repairs}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
