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

from vlmeval.vlm.qwen2_vl.model import Qwen2VLChatReplay
from vlmeval.vlm.qwen2_vl.replay_prompt_template import PROMPT_TEMPLATE_IDENTITY
from vlmeval.vlm.replay_policy import (
    REPLAY_IMAGE_TEXT,
    REPLAY_TEXT_IMAGE,
    REPLAY_IMAGE_TEXT_IMAGE_TEXT,
    REPLAY_IMAGE_TEXT_IMAGE,
    REPLAY_IMAGE_TEXT_TEXT,
    REPLAY_IMAGE_IMAGE_TEXT,
)


def compact_text(s: str, max_len: int = 80) -> str:
    s = s.replace("\n", "\\n")
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def summarize_content(content: list[dict[str, Any]]) -> dict[str, Any]:
    order = [item.get("type", "unknown") for item in content]
    counts = {"text": 0, "image": 0, "video": 0}
    text_snippets = []
    image_refs = []
    for item in content:
        t = item.get("type")
        if t in counts:
            counts[t] += 1
        if t == "text":
            text_snippets.append(compact_text(item.get("text", "")))
        elif t == "image":
            image_refs.append(item.get("image", ""))
        elif t == "video":
            image_refs.append(item.get("video", ""))
    return {
        "counts": counts,
        "order": order,
        "text_preview": text_snippets,
        "vision_preview": image_refs,
    }


class ReplayPrepareContentProbe(Qwen2VLChatReplay):
    """Lightweight probe to test prepare_content replay behavior without loading the model."""

    def __init__(self, replay_mode: str, repeat_times: int, limit_mm_per_prompt: int):
        # Do not call parent __init__ (it would load the full model).
        self.replay_cfg = {
            "mode": replay_mode,
            "repeat_times": repeat_times,
            "debug": False,
            "image_copy_mode": "reuse_path",
        }
        self.prompt_template_cfg = {
            "name": PROMPT_TEMPLATE_IDENTITY,
            "template": "{problem}",
            "source": "probe",
        }
        self.min_pixels = None
        self.max_pixels = None
        self.total_pixels = None
        self.fps = 2
        self.nframe = 128
        self.FRAME_FACTOR = 2
        self.limit_mm_per_prompt = limit_mm_per_prompt
        self.use_audio_in_video = False
        self.verbose = False
        self.model_path = "probe-only"
        self.system_prompt = None
        self.post_process = False
        self._stage_debug_active = False


def build_test_input(image_path: str) -> list[dict[str, str]]:
    return [
        {"type": "image", "value": image_path},
        {"type": "text", "value": "Question: Which option is correct? A. cat B. dog"},
    ]


def run_single_mode(mode: str, image_path: str, dataset: str, repeat_times: int, limit_mm_per_prompt: int) -> dict[str, Any]:
    probe = ReplayPrepareContentProbe(
        replay_mode=mode,
        repeat_times=repeat_times,
        limit_mm_per_prompt=limit_mm_per_prompt,
    )
    inputs = build_test_input(image_path)
    content_for_transformers = probe._prepare_content(inputs, dataset=dataset)
    content_for_vllm = probe._prepare_content_vllm(inputs, dataset=dataset)
    return {
        "mode": mode,
        "input": inputs,
        "prepare_content": summarize_content(content_for_transformers),
        "prepare_content_vllm": summarize_content(content_for_vllm),
        "prepare_content_raw": content_for_transformers,
        "prepare_content_vllm_raw": content_for_vllm,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check replay effects on _prepare_content and _prepare_content_vllm.")
    parser.add_argument(
        "--image-path",
        type=str,
        default="image_test.png",
        help="Image path used in the test prompt.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MMBench_DEV_EN_V11",
        help="Dataset name passed into prepare_content.",
    )
    parser.add_argument(
        "--repeat-times",
        type=int,
        default=1,
        help="Replay repeat times.",
    )
    parser.add_argument(
        "--limit-mm-per-prompt",
        type=int,
        default=8,
        help="limit_mm_per_prompt used in _prepare_content_vllm.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image_path
    if not os.path.isabs(image_path):
        image_path = os.path.abspath(image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path not found: {image_path}")

    modes = [REPLAY_IMAGE_TEXT, REPLAY_TEXT_IMAGE, REPLAY_IMAGE_TEXT_TEXT, REPLAY_IMAGE_TEXT_IMAGE, REPLAY_IMAGE_TEXT_IMAGE_TEXT, REPLAY_IMAGE_IMAGE_TEXT]
    results = []
    for mode in modes:
        results.append(
            run_single_mode(
                mode=mode,
                image_path=image_path,
                dataset=args.dataset,
                repeat_times=args.repeat_times,
                limit_mm_per_prompt=args.limit_mm_per_prompt,
            )
        )

    print(json.dumps({"image_path": image_path, "dataset": args.dataset, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
