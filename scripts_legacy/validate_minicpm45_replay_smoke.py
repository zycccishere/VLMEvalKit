#!/usr/bin/env python3
import argparse
import copy
import json
import os
import sys
import traceback
from typing import Any


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


DEFAULT_DATASETS = [
    "AI2D_TEST",
    "DynaMath",
    "MathVision",
    "MathVista_MINI",
    "OCRBench",
    "SEEDBench2_Plus",
    "VisuLogic",
    "LogicVista",
    "VisualPuzzles",
]

DEFAULT_REPLAY_MODES = [
    "image_text",
    "text_image",
    "image_text_text",
    "image_text_image",
    "image_text_image_text",
    "image_image_text",
]


def compact_text(text: str, limit: int = 240) -> str:
    text = str(text).replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + f"...[+{len(text) - limit} chars]"


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def jsonable_message(message: list[dict[str, Any]], text_limit: int = 320) -> list[dict[str, Any]]:
    out = []
    for item in message:
        item_type = item.get("type", "unknown")
        value = item.get("value", item.get("text", ""))
        if item_type == "text":
            out.append({"type": "text", "value": compact_text(str(value), text_limit)})
        else:
            out.append({"type": item_type, "value": os.path.basename(str(value))})
    return out


def summarize_message(message: list[dict[str, Any]], text_limit: int = 240) -> dict[str, Any]:
    texts = []
    order = []
    for item in message:
        order.append(item.get("type", "unknown"))
        if item.get("type") == "text":
            texts.append(compact_text(str(item.get("value", item.get("text", ""))), text_limit))
    return {
        "order": order,
        "num_images": sum(1 for item in message if item.get("type") == "image"),
        "num_texts": sum(1 for item in message if item.get("type") == "text"),
        "texts": texts,
    }


def output_metrics(output: str, policy: str, dataset_type: str) -> dict[str, Any]:
    stripped = str(output).strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    metrics = {
        "nonempty": bool(stripped),
        "char_count": len(stripped),
        "line_count": len(lines),
        "contains_answer_prefix": "answer:" in stripped.lower(),
        "contains_think": "<think>" in stripped.lower(),
        "preview": compact_text(stripped, 280),
    }
    if policy == "directly_answer":
        metrics["short_like"] = bool(stripped) and len(stripped) <= 256 and len(lines) <= 3
        if dataset_type == "MCQ":
            metrics["mcq_letter_like"] = stripped in list("ABCDEFGHI")
    return metrics


def expected_order_for_single_image_prompt(mode: str) -> list[str]:
    mapping = {
        "image_text": ["image", "text"],
        "text_image": ["text", "image"],
        "image_text_text": ["image", "text", "text"],
        "image_text_image": ["image", "text", "image"],
        "image_text_image_text": ["image", "text", "image", "text"],
        "image_image_text": ["image", "image", "text"],
    }
    return mapping[mode]


def validate_replayed_input(
    base_message: list[dict[str, Any]],
    replayed_message: list[dict[str, Any]],
    mode: str,
    policy: str,
    dataset: str,
    prompt_template_cfg: dict[str, str],
    render_prompt_with_template,
) -> dict[str, Any]:
    base_summary = summarize_message(base_message)
    replay_summary = summarize_message(replayed_message)
    checks: dict[str, Any] = {
        "base_summary": base_summary,
        "replay_summary": replay_summary,
    }

    if base_summary["num_images"] == 1 and base_summary["num_texts"] == 1:
        expected_order = expected_order_for_single_image_prompt(mode)
        checks["expected_order"] = expected_order
        checks["order_ok"] = replay_summary["order"] == expected_order
    else:
        checks["expected_order"] = None
        checks["order_ok"] = None

    if policy == "directly_answer":
        base_texts = [item.get("value", "") for item in base_message if item.get("type") == "text"]
        replay_texts = [item.get("value", "") for item in replayed_message if item.get("type") == "text"]
        base_last_text = base_texts[-1] if base_texts else ""
        rendered = render_prompt_with_template(base_last_text, prompt_template_cfg, dataset=dataset)
        checks["template_render_preview"] = compact_text(rendered, 240)
        checks["last_text_rendered_ok"] = bool(replay_texts) and replay_texts[-1] == rendered
        checks["rendered_text_count"] = sum(text == rendered for text in replay_texts)
        checks["rendered_text_only_at_last"] = (
            bool(replay_texts)
            and replay_texts[-1] == rendered
            and all(text != rendered for text in replay_texts[:-1])
        )
    else:
        checks["template_render_preview"] = None
        checks["last_text_rendered_ok"] = None
        checks["rendered_text_count"] = None
        checks["rendered_text_only_at_last"] = None

    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 2-sample replay smoke checks for MiniCPM-V-4_5-Replay across datasets and modes."
    )
    parser.add_argument("--model-path", type=str, default="/models/MiniCPM-V-4_5")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--replay-modes", nargs="*", default=DEFAULT_REPLAY_MODES)
    parser.add_argument("--policies", nargs="*", default=["identity", "directly_answer"])
    parser.add_argument("--samples-per-dataset", type=int, default=2)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--template-on-last-replay-text", type=int, default=1)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--output-summary", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from vlmeval.dataset import DATASET_TYPE, build_dataset
    from vlmeval.vlm.minicpm_v_4_5_replay import MiniCPM_V_4_5_Replay
    from vlmeval.vlm.qwen2_vl.replay_prompt_template import (
        read_prompt_template_config_from_env,
        render_prompt_with_template,
    )

    os.environ.setdefault("MINICPM_FORCE_NO_THINKING", "1")
    os.environ.setdefault("REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT", str(args.template_on_last_replay_text))
    os.environ.pop("REPLAY_PROMPT_TEMPLATE_FILE", None)
    os.environ.pop("REPLAY_PROMPT_TEMPLATE", None)

    model = MiniCPM_V_4_5_Replay(model_path=args.model_path, use_vllm=args.use_vllm)

    all_records: list[dict[str, Any]] = []

    for policy in args.policies:
        os.environ["REPLAY_PROMPT_TEMPLATE_NAME"] = policy
        prompt_template_cfg = read_prompt_template_config_from_env()
        model.prompt_template_cfg = prompt_template_cfg
        model.template_on_last_replay_text = bool(args.template_on_last_replay_text)

        for dataset_name in args.datasets:
            dataset = build_dataset(dataset_name)
            if dataset is None:
                all_records.append(
                    {
                        "policy": policy,
                        "dataset": dataset_name,
                        "status": "dataset_build_failed",
                    }
                )
                continue

            model.set_dump_image(dataset.dump_image)
            total = len(dataset.data)
            upper = min(total, args.start_offset + args.samples_per_dataset)

            for mode in args.replay_modes:
                model.replay_cfg = {
                    "mode": mode,
                    "repeat_times": 1,
                    "debug": False,
                    "image_copy_mode": "reuse_path",
                }

                for row_pos in range(args.start_offset, upper):
                    item = dataset.data.iloc[row_pos]
                    sample_index = item["index"]
                    dataset_type = DATASET_TYPE(dataset_name)

                    try:
                        base_message = model.build_prompt(item, dataset=dataset_name)
                        replayed_message = model._apply_replay_pipeline(copy.deepcopy(base_message), dataset=dataset_name)
                        input_checks = validate_replayed_input(
                            base_message=base_message,
                            replayed_message=replayed_message,
                            mode=mode,
                            policy=policy,
                            dataset=dataset_name,
                            prompt_template_cfg=prompt_template_cfg,
                            render_prompt_with_template=render_prompt_with_template,
                        )
                        output = model.generate_inner(copy.deepcopy(base_message), dataset=dataset_name)
                        metrics = output_metrics(output, policy=policy, dataset_type=dataset_type)
                        record = {
                            "status": "ok",
                            "policy": policy,
                            "dataset": dataset_name,
                            "dataset_type": dataset_type,
                            "replay_mode": mode,
                            "row_pos": row_pos,
                            "sample_index": sample_index,
                            "input_checks": input_checks,
                            "base_message": jsonable_message(base_message),
                            "replayed_message": jsonable_message(replayed_message),
                            "output_metrics": metrics,
                            "output": output,
                        }
                    except Exception as err:
                        record = {
                            "status": "error",
                            "policy": policy,
                            "dataset": dataset_name,
                            "dataset_type": dataset_type,
                            "replay_mode": mode,
                            "row_pos": row_pos,
                            "sample_index": sample_index,
                            "error_type": type(err).__name__,
                            "error": str(err),
                            "traceback": traceback.format_exc(),
                        }

                    all_records.append(record)
                    print(
                        json.dumps(
                            {
                                "policy": policy,
                                "dataset": dataset_name,
                                "mode": mode,
                                "row_pos": row_pos,
                                "status": record["status"],
                                "preview": record.get("output_metrics", {}).get("preview", ""),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")

    ok_records = [record for record in all_records if record.get("status") == "ok"]
    error_records = [record for record in all_records if record.get("status") == "error"]
    summary = {
        "total_records": len(all_records),
        "ok_records": len(ok_records),
        "error_records": len(error_records),
        "datasets": args.datasets,
        "replay_modes": args.replay_modes,
        "policies": args.policies,
        "samples_per_dataset": args.samples_per_dataset,
        "template_on_last_replay_text": bool(args.template_on_last_replay_text),
        "use_vllm": bool(args.use_vllm),
    }

    for policy in args.policies:
        policy_records = [record for record in ok_records if record["policy"] == policy]
        policy_key = policy.replace("-", "_")
        summary[f"{policy_key}_records"] = len(policy_records)
        summary[f"{policy_key}_order_ok"] = sum(
            1 for record in policy_records if record["input_checks"].get("order_ok") is True
        )
        summary[f"{policy_key}_nonempty"] = sum(
            1 for record in policy_records if record["output_metrics"].get("nonempty")
        )
        if policy == "directly_answer":
            summary["directly_answer_template_last_only_ok"] = sum(
                1
                for record in policy_records
                if record["input_checks"].get("rendered_text_only_at_last") is True
            )
            summary["directly_answer_short_like"] = sum(
                1 for record in policy_records if record["output_metrics"].get("short_like")
            )
            summary["directly_answer_mcq_letter_like"] = sum(
                1 for record in policy_records if record["output_metrics"].get("mcq_letter_like") is True
            )

    os.makedirs(os.path.dirname(args.output_summary), exist_ok=True)
    with open(args.output_summary, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, ensure_ascii=False, indent=2)

    print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
