#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


TARGET_INDEX = "785"
GENERATED_COLUMNS = ("prediction", "description", "detailed_prediction", "full_output")
INVALID_OUTPUTS = {"", "none", "nan", "null"}


def normalized(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def row_for_index(frame: pd.DataFrame, index: str) -> pd.Series:
    matches = frame[frame["index"].astype(str) == index]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row for index={index}, found {len(matches)}")
    return matches.iloc[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-xlsx", required=True)
    parser.add_argument("--canonical-manifest", required=True)
    parser.add_argument("--repair-xlsx", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    canonical_path = Path(args.canonical_xlsx).resolve()
    manifest_path = Path(args.canonical_manifest).resolve()
    repair_path = Path(args.repair_xlsx).resolve()
    report_path = Path(args.report).resolve()
    canonical = pd.read_excel(canonical_path, keep_default_na=False)
    repair = pd.read_excel(repair_path, keep_default_na=False)

    if len(canonical) != 1000 or canonical["index"].astype(str).nunique() != 1000:
        raise ValueError("Canonical OCRBench table must contain 1000 unique indices")
    if len(repair) != 1 or repair["index"].astype(str).tolist() != [TARGET_INDEX]:
        raise ValueError("Repair table must contain only index 785")

    before = row_for_index(canonical, TARGET_INDEX)
    replacement = row_for_index(repair, TARGET_INDEX)
    replacement_values = {column: normalized(replacement.get(column, "")) for column in GENERATED_COLUMNS}
    if replacement_values["prediction"].lower() in INVALID_OUTPUTS:
        raise ValueError(f"Repair prediction is invalid: {replacement_values['prediction']!r}")

    backup_dir = report_path.parent / "repair_backup_20260728"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / canonical_path.name
    if not backup_path.exists():
        shutil.copy2(canonical_path, backup_path)

    before_non_target = canonical[canonical["index"].astype(str) != TARGET_INDEX].copy()
    target_mask = canonical["index"].astype(str) == TARGET_INDEX
    for column, value in replacement_values.items():
        if column not in canonical.columns:
            canonical[column] = ""
        canonical.loc[target_mask, column] = value

    temp_path = canonical_path.with_suffix(".repairing.xlsx")
    canonical.to_excel(temp_path, index=False)
    temp_path.replace(canonical_path)
    merged = pd.read_excel(canonical_path, keep_default_na=False)
    after_non_target = merged[merged["index"].astype(str) != TARGET_INDEX].copy()
    after = row_for_index(merged, TARGET_INDEX)

    non_target_unchanged = before_non_target.reset_index(drop=True).equals(after_non_target.reset_index(drop=True))
    all_predictions_valid = all(
        normalized(value).lower() not in INVALID_OUTPUTS for value in merged["prediction"].tolist()
    )
    checks = {
        "repair_single_target": True,
        "repair_prediction_valid": replacement_values["prediction"].lower() not in INVALID_OUTPUTS,
        "canonical_rows_1000": len(merged) == 1000,
        "canonical_indices_unique": merged["index"].astype(str).nunique() == 1000,
        "non_target_rows_unchanged": non_target_unchanged,
        "all_predictions_valid": all_predictions_valid,
    }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["updated_at"] = datetime.now().strftime("%Y%m%d%H%M%S")
    manifest["repair"] = {
        "target_index": int(TARGET_INDEX),
        "source_prediction_file": str(repair_path),
        "backup_prediction_file": str(backup_path),
        "merge_report": str(report_path),
        "checks": checks,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "target_index": int(TARGET_INDEX),
        "canonical_prediction_file": str(canonical_path),
        "repair_prediction_file": str(repair_path),
        "backup_prediction_file": str(backup_path),
        "before": {column: normalized(before.get(column, "")) for column in GENERATED_COLUMNS},
        "after": {column: normalized(after.get(column, "")) for column in GENERATED_COLUMNS},
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    print(json.dumps({"all_passed": report["all_passed"], "after_prediction": report["after"]["prediction"]}))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
