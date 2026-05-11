#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vlmeval.dataset import build_dataset  # noqa: E402
from vlmeval.vlm.qwen2_vl.replay_prompt_template import (  # noqa: E402
    PROMPT_TEMPLATE_DIRECTLY_ANSWER,
    PROMPT_TEMPLATE_IDENTITY,
    apply_prompt_template_to_content,
)
from vlmeval.vlm.replay_policy import apply_replay, canonicalize_replay_mode, is_noop_replay_mode  # noqa: E402


def _to_content(message: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _extract_text_blocks(content: list[dict[str, Any]]) -> list[str]:
    outs = []
    for item in content:
        if item.get("type") == "text":
            outs.append(str(item.get("text", "")))
    return outs


def _apply_pipeline(
    content: list[dict[str, Any]],
    template_name: str,
    replay_mode: str,
    replay_times: int,
    template_on_last_replay_text: bool,
) -> list[dict[str, Any]]:
    template_map = {
        PROMPT_TEMPLATE_IDENTITY: "{problem}",
        PROMPT_TEMPLATE_DIRECTLY_ANSWER: (
            "{problem}\n"
            "Return exactly **one line** in this format: <ANSWER>\n"
            "Do not output any explanation, derivation, words, or extra symbols."
        ),
    }
    template_text = template_map.get(template_name, template_name)
    template_cfg = {"name": template_name, "template": template_text, "source": "check_script"}

    replay_mode = canonicalize_replay_mode(replay_mode)
    if template_on_last_replay_text and not is_noop_replay_mode(replay_mode):
        replayed = apply_replay(
            content,
            mode=replay_mode,
            repeat_times=replay_times,
            image_copy_mode="reuse_path",
        )
        return apply_prompt_template_to_content(replayed, template_cfg)

    templated = apply_prompt_template_to_content(content, template_cfg)
    return apply_replay(
        templated,
        mode=replay_mode,
        repeat_times=replay_times,
        image_copy_mode="reuse_path",
    )


def _pick_line(dataset_obj, row_id: int, sample_index: str | None):
    data = dataset_obj.data
    if sample_index is not None:
        sub = data[data["index"].astype(str) == str(sample_index)]
        if len(sub) == 0:
            raise ValueError(f"index={sample_index} not found in dataset.")
        return sub.iloc[0]
    if row_id < 0 or row_id >= len(data):
        raise ValueError(f"row_id={row_id} out of range, dataset size={len(data)}")
    return data.iloc[row_id]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect final prompt after dataset default prompt + direct template (+ optional replay), no model forward."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. VisuLogic / LogicVista / VisualPuzzles")
    parser.add_argument("--row-id", type=int, default=0, help="Row offset in dataset.data")
    parser.add_argument("--sample-index", type=str, default=None, help="Exact value in dataset index column")
    parser.add_argument("--template-name", type=str, default=PROMPT_TEMPLATE_DIRECTLY_ANSWER, help="identity / directly_answer")
    parser.add_argument("--replay-mode", type=str, default="image_text", help="image_text / text_image / image_text_text / image_text_image / image_text_image_text / image_image_text (legacy alias: none)")
    parser.add_argument("--replay-times", type=int, default=1)
    parser.add_argument("--template-on-last-replay-text", action="store_true")
    parser.add_argument("--print-full-content", action="store_true", help="Print full JSON content including image/video refs")
    return parser.parse_args()


def main():
    args = parse_args()

    ds = build_dataset(args.dataset)
    if ds is None:
        raise RuntimeError(f"Failed to build dataset: {args.dataset}")

    line = _pick_line(ds, args.row_id, args.sample_index)
    raw_message = ds.build_prompt(line)
    raw_content = _to_content(raw_message)
    final_content = _apply_pipeline(
        raw_content,
        template_name=args.template_name,
        replay_mode=args.replay_mode,
        replay_times=args.replay_times,
        template_on_last_replay_text=args.template_on_last_replay_text,
    )

    raw_text_blocks = _extract_text_blocks(raw_content)
    final_text_blocks = _extract_text_blocks(final_content)

    result = {
        "dataset": args.dataset,
        "row_id": args.row_id,
        "sample_index": str(line["index"]) if "index" in line else None,
        "template_name": args.template_name,
        "replay_mode": args.replay_mode,
        "replay_times": args.replay_times,
        "template_on_last_replay_text": args.template_on_last_replay_text,
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
        "raw_text_blocks": raw_text_blocks,
        "final_text_blocks": final_text_blocks,
    }

    if args.print_full_content:
        result["raw_content"] = raw_content
        result["final_content"] = final_content

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
