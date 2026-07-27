#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODELS = ("qwen25vl_32b", "qwen25vl_3b", "minicpm_o_45")
DATASETS = (
    "MathVision",
    "DynaMath",
    "LogicVista",
    "VisualPuzzles",
    "AI2D_TEST",
    "OCRBench",
    "SEEDBench2_Plus",
)
EXPECTED_MODE = "image_text_image_text"
EXPECTED_TRANSFORM = "shift_right_fixed_28px"
FAILURE_MARKERS = ("[FAILED_INFER]", "Failed to obtain answer via API.")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def nonblank_outputs(frame: pd.DataFrame) -> list[str]:
    columns = [name for name in ("prediction", "detailed_prediction", "description") if name in frame.columns]
    outputs = []
    for _, row in frame.iterrows():
        value = ""
        for column in columns:
            candidate = row[column]
            if pd.notna(candidate) and str(candidate).strip():
                value = str(candidate).strip()
                break
        outputs.append(value)
    return outputs


def validate_trace(trace_file: Path, model: str) -> dict[str, bool | int | str]:
    records = load_jsonl(trace_file)
    transforms = [item for item in records if item.get("phase") == "image_transform"]
    checks: dict[str, bool | int | str] = {
        "trace_file": str(trace_file),
        "transform_records": len(transforms),
        "has_transform": bool(transforms),
    }
    checks["all_transforms_match"] = bool(transforms) and all(
        item.get("image_transform") == EXPECTED_TRANSFORM
        and item.get("record", {}).get("target_image_position") == 2
        and item.get("record", {}).get("content_item_index") == 2
        and item.get("record", {}).get("shift", {}).get("dx") == 28
        and item.get("record", {}).get("shift", {}).get("dy") == 0
        and item.get("record", {}).get("shift", {}).get("semantic_unit") == "fixed_processed_pixels"
        and item.get("record", {}).get("shift", {}).get("border_wrap_verified") is True
        for item in transforms
    )

    if model.startswith("qwen"):
        prepared = [item for item in records if item.get("phase") == "prepared"]
        item = prepared[0] if prepared else {}
        replayed = item.get("message_replayed", [])
        spans = item.get("image_token_spans", [])
        checks.update(
            {
                "has_prepared": bool(prepared),
                "iqiq_order": [part.get("type") for part in replayed] == ["image", "text", "image", "text"],
                "duplicated_question": bool(replayed) and replayed[1].get("text") == replayed[3].get("text"),
                "two_images": item.get("vision_extract_image_count") == 2,
                "two_spans": [span.get("image_position") for span in spans] == [1, 2],
                "target_span_i2": item.get("target_image_span", {}).get("image_position") == 2,
            }
        )
    else:
        payloads = [item for item in records if item.get("phase") == "minicpm_vllm_payload"]
        item = payloads[0] if payloads else {}
        replayed = item.get("message_replayed", [])
        checks.update(
            {
                "has_payload": bool(payloads),
                "iqiq_order": [part.get("type") for part in replayed] == ["image", "text", "image", "text"],
                "duplicated_question": bool(replayed) and replayed[1].get("value") == replayed[3].get("value"),
                "two_images": item.get("payload_image_count") == 2,
                "second_image_transformed": bool(replayed)
                and "_shift_right_fixed_28px.png" in str(replayed[2].get("value", "")),
            }
        )
    return checks


def bool_checks_pass(checks: dict) -> bool:
    return all(value for value in checks.values() if isinstance(value, bool))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    task_results = []

    for model in MODELS:
        for dataset in DATASETS:
            task_root = run_root / "default" / EXPECTED_MODE / EXPECTED_TRANSFORM / model / dataset
            manifests = list((task_root / "predictions").glob("manifest.json"))
            checks: dict[str, bool | int | float | str] = {"single_manifest": len(manifests) == 1}
            if len(manifests) != 1:
                task_results.append({"model": model, "dataset": dataset, "checks": checks, "all_passed": False})
                continue
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            prediction_file = Path(manifest.get("prediction_file", ""))
            # OCR/VQA models may legitimately answer with the literal string
            # "None". Preserve strings instead of letting pandas coerce them
            # into missing values.
            frame = pd.read_excel(prediction_file, keep_default_na=False)
            expected_rows = int(manifest.get("expected_rows", -1))
            outputs = nonblank_outputs(frame)
            index_values = frame["index"].astype(str).tolist() if "index" in frame.columns else []
            combined = "\n".join(outputs)
            checks.update(
                {
                    "manifest_complete": manifest.get("status") == "complete",
                    "mode_matches": manifest.get("task", {}).get("mode") == EXPECTED_MODE,
                    "transform_matches": manifest.get("task", {}).get("transform") == EXPECTED_TRANSFORM,
                    "model_matches": manifest.get("task", {}).get("model_key") == model,
                    "dataset_matches": manifest.get("task", {}).get("dataset") == dataset,
                    "prediction_exists": prediction_file.is_file(),
                    "row_count": len(frame),
                    "expected_rows": expected_rows,
                    "rows_match": len(frame) == expected_rows,
                    "index_present": bool(index_values),
                    "index_unique": len(index_values) == len(set(index_values)),
                    "outputs_nonblank": len(outputs) == len(frame) and all(outputs),
                    "distinct_outputs": len(set(outputs)),
                    "output_not_collapsed": len(set(outputs)) > 1,
                    "no_failure_markers": not any(marker in combined for marker in FAILURE_MARKERS),
                }
            )

            trace_files = list((task_root / "_trace").glob("*.jsonl"))
            checks["single_trace_file"] = len(trace_files) == 1
            trace_checks = validate_trace(trace_files[0], model) if len(trace_files) == 1 else {}
            task_results.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "prediction_file": str(prediction_file),
                    "checks": checks,
                    "trace_checks": trace_checks,
                    "sample_outputs": outputs[:3],
                    "all_passed": bool_checks_pass(checks) and bool_checks_pass(trace_checks),
                }
            )

    summary = {
        "run_root": str(run_root),
        "contract": "IQIQ; I2-only; circular processed-image shift right by fixed 28 pixels",
        "expected_tasks": len(MODELS) * len(DATASETS),
        "validated_tasks": len(task_results),
        "tasks": task_results,
        "all_passed": len(task_results) == len(MODELS) * len(DATASETS)
        and all(item["all_passed"] for item in task_results),
    }
    output_dir = run_root / "_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "full_inference_validation.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    print(
        json.dumps(
            {
                "all_passed": summary["all_passed"],
                "validated_tasks": summary["validated_tasks"],
                "failed_tasks": [
                    f"{item['model']}/{item['dataset']}" for item in task_results if not item["all_passed"]
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
