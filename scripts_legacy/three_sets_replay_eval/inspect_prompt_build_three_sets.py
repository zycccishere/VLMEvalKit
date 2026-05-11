#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vlmeval.dataset import build_dataset  # noqa: E402
from vlmeval.vlm.qwen2_vl.replay_prompt_template import (  # noqa: E402
    PROMPT_TEMPLATE_DIRECTLY_ANSWER,
    PROMPT_TEMPLATE_IDENTITY,
    apply_prompt_template_to_content,
    strip_prompt_template_from_content_for_direct_answer,
)
from vlmeval.vlm.replay_policy import apply_replay  # noqa: E402


DEFAULT_DATASETS = ["VisuLogic", "LogicVista", "VisualPuzzles"]
DEFAULT_REPLAY_MODES = [
    "none",
    "image_text_text",
    "image_text_image",
    "image_text_image_text",
    "image_image_text",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect prompt construction before model forward on the three repaired datasets. "
            "No model inference is performed."
        )
    )
    parser.add_argument(
        "--setting",
        choices=["default", "direct"],
        default="direct",
        help="default = dataset-native prompt only; direct = replay then apply directly_answer on last replay text.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to inspect.",
    )
    parser.add_argument(
        "--replay-modes",
        nargs="+",
        default=DEFAULT_REPLAY_MODES,
        help="Replay modes to inspect.",
    )
    parser.add_argument(
        "--rows",
        nargs="+",
        type=int,
        default=[0],
        help="Row offsets in dataset.data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to dump one JSON per (dataset, replay_mode, row).",
    )
    parser.add_argument(
        "--print-full-content",
        action="store_true",
        help="Include full multimodal content payloads in output.",
    )
    return parser.parse_args()


def to_content(message: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content = []
    for item in message:
        t = item.get("type")
        v = item.get("value")
        if t == "text":
            content.append({"type": "text", "text": str(v)})
        elif t == "image":
            content.append({"type": "image", "image": str(v)})
        elif t == "video":
            content.append({"type": "video", "video": str(v)})
        else:
            content.append({"type": str(t), "value": v})
    return content


def extract_text_blocks(content: list[dict[str, Any]]) -> list[str]:
    outs = []
    for item in content:
        if item.get("type") == "text":
            outs.append(str(item.get("text", "")))
    return outs


def template_cfg_for_setting(setting: str) -> dict[str, str]:
    if setting == "default":
        return {
            "name": PROMPT_TEMPLATE_IDENTITY,
            "template": "{problem}",
            "source": "three_sets_verify_default",
        }
    return {
        "name": PROMPT_TEMPLATE_DIRECTLY_ANSWER,
        "template": (
            "{problem}\n"
            "Answer directly with a single word or short phrase.\n"
            "Do not output any explanation, derivation, words, or extra symbols."
        ),
        "source": "three_sets_verify_direct",
    }


def apply_pipeline(
    content: list[dict[str, Any]],
    dataset_name: str,
    setting: str,
    replay_mode: str,
    replay_times: int = 1,
) -> list[dict[str, Any]]:
    template_cfg = template_cfg_for_setting(setting)
    template_on_last_replay_text = True

    if template_on_last_replay_text and replay_mode != "none":
        replay_source = content
        if setting == "direct":
            replay_source = strip_prompt_template_from_content_for_direct_answer(
                content,
                dataset=dataset_name,
                text_key="text",
            )
        replayed = apply_replay(
            replay_source,
            mode=replay_mode,
            repeat_times=replay_times,
            image_copy_mode="reuse_path",
        )
        return apply_prompt_template_to_content(replayed, template_cfg, dataset=dataset_name)

    templated = apply_prompt_template_to_content(content, template_cfg, dataset=dataset_name)
    return apply_replay(
        templated,
        mode=replay_mode,
        repeat_times=replay_times,
        image_copy_mode="reuse_path",
    )


def pick_line(dataset_obj, row_id: int):
    data = dataset_obj.data
    if row_id < 0 or row_id >= len(data):
        raise ValueError(f"row_id={row_id} out of range, dataset size={len(data)}")
    return data.iloc[row_id]


def dump_result(path: str, payload: dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    all_results = []

    for dataset_name in args.datasets:
        ds = build_dataset(dataset_name)
        if ds is None:
            raise RuntimeError(f"Failed to build dataset: {dataset_name}")

        for row_id in args.rows:
            line = pick_line(ds, row_id)
            raw_message = ds.build_prompt(line)
            raw_content = to_content(raw_message)

            for replay_mode in args.replay_modes:
                final_content = apply_pipeline(
                    raw_content,
                    dataset_name=dataset_name,
                    setting=args.setting,
                    replay_mode=replay_mode,
                    replay_times=1,
                )

                result = {
                    "dataset": dataset_name,
                    "row_id": row_id,
                    "sample_index": str(line["index"]) if "index" in line else None,
                    "setting": args.setting,
                    "replay_mode": replay_mode,
                    "raw_counts": {
                        "text": sum(1 for x in raw_content if x.get("type") == "text"),
                        "image": sum(1 for x in raw_content if x.get("type") == "image"),
                        "video": sum(1 for x in raw_content if x.get("type") == "video"),
                    },
                    "final_counts": {
                        "text": sum(1 for x in final_content if x.get("type") == "text"),
                        "image": sum(1 for x in final_content if x.get("type") == "image"),
                        "video": sum(1 for x in final_content if x.get("type") == "video"),
                    },
                    "raw_text_blocks": extract_text_blocks(raw_content),
                    "final_text_blocks": extract_text_blocks(final_content),
                }

                if args.print_full_content:
                    result["raw_content"] = raw_content
                    result["final_content"] = final_content

                all_results.append(result)

                if args.output_dir:
                    out_name = f"{dataset_name}__{args.setting}__{replay_mode}__row{row_id}.json"
                    dump_result(os.path.join(args.output_dir, out_name), result)

    print(json.dumps(all_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
