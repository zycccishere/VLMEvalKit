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
from vlmeval.vlm.qwen35_vl_replay import Qwen35VLChatReplay
from vlmeval.vlm.minicpm_v import MiniCPM_V_2_6_Replay
from vlmeval.vlm.minicpm_v_4_5_replay import MiniCPM_V_4_5_Replay
from vlmeval.vlm.qwen2_vl.replay_prompt_template import (
    PROMPT_TEMPLATE_DIRECTLY_ANSWER,
    PROMPT_TEMPLATE_IDENTITY,
)
from vlmeval.vlm.replay_policy import (
    REPLAY_IMAGE_TEXT,
    REPLAY_TEXT_IMAGE,
    REPLAY_IMAGE_TEXT_TEXT,
    REPLAY_IMAGE_TEXT_IMAGE,
    REPLAY_IMAGE_TEXT_IMAGE_TEXT,
    REPLAY_IMAGE_IMAGE_TEXT,
)


def compact_text(s: str, max_len: int = 120) -> str:
    s = s.replace("\n", "\\n")
    return s if len(s) <= max_len else s[:max_len] + "..."


def summarize_content(content: list[dict[str, Any]]) -> dict[str, Any]:
    order = [item.get("type", "unknown") for item in content]
    texts = []
    for item in content:
        if item.get("type") == "text":
            texts.append(compact_text(str(item.get("text", item.get("value", "")))))
    return {
        "order": order,
        "text_preview": texts,
        "raw": content,
    }


def base_template_cfg(name: str) -> dict[str, str]:
    if name == PROMPT_TEMPLATE_DIRECTLY_ANSWER:
        template = "{problem}\nAnswer the question using a single word or phrase."
    else:
        template = "{problem}"
    return {"name": name, "template": template, "source": "probe"}


class Qwen2Probe(Qwen2VLChatReplay):
    def __init__(self, replay_mode: str, template_name: str, template_on_last_replay_text: bool, limit_mm_per_prompt: int):
        self.replay_cfg = {"mode": replay_mode, "repeat_times": 1, "debug": False, "image_copy_mode": "reuse_path"}
        self.prompt_template_cfg = base_template_cfg(template_name)
        self.template_on_last_replay_text = template_on_last_replay_text
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
        self._stage_debug_enabled = False
        self._stage_debug_max_samples = 0
        self._stage_debug_seen_samples = 0
        self._stage_debug_active = False
        self._replay_dump_dir = ""
        self._replay_dump_file = None
        self._replay_dump_max_chars = 0
        self._prompt_audit_enabled = False
        self._prompt_audit_print = False
        self.safe_fallback_enabled = False


class Qwen35Probe(Qwen35VLChatReplay):
    def __init__(self, replay_mode: str, template_name: str, template_on_last_replay_text: bool, limit_mm_per_prompt: int):
        self.replay_cfg = {"mode": replay_mode, "repeat_times": 1, "debug": False, "image_copy_mode": "reuse_path"}
        self.prompt_template_cfg = base_template_cfg(template_name)
        self.template_on_last_replay_text = template_on_last_replay_text
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
        self._stage_debug_enabled = False
        self._stage_debug_max_samples = 0
        self._stage_debug_seen_samples = 0
        self._stage_debug_active = False
        self._replay_dump_dir = ""
        self._replay_dump_file = None
        self._replay_dump_max_chars = 0
        self._prompt_audit_enabled = False
        self._prompt_audit_print = False
        self.safe_fallback_enabled = False


class MiniCPM26Probe(MiniCPM_V_2_6_Replay):
    def __init__(self, replay_mode: str, template_name: str, template_on_last_replay_text: bool):
        self.replay_cfg = {"mode": replay_mode, "repeat_times": 1, "debug": False, "image_copy_mode": "reuse_path"}
        self.prompt_template_cfg = base_template_cfg(template_name)
        self.template_on_last_replay_text = template_on_last_replay_text


class MiniCPM45Probe(MiniCPM_V_4_5_Replay):
    def __init__(self, replay_mode: str, template_name: str, template_on_last_replay_text: bool):
        self.replay_cfg = {"mode": replay_mode, "repeat_times": 1, "debug": False, "image_copy_mode": "reuse_path"}
        self.prompt_template_cfg = base_template_cfg(template_name)
        self.template_on_last_replay_text = template_on_last_replay_text


def build_test_input(image_path: str) -> list[dict[str, str]]:
    return [
        {"type": "image", "value": image_path},
        {"type": "text", "value": "Question: Which option is correct? A. cat B. dog"},
    ]


def prepare_minicpm_message(message: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"type": item["type"], "value": item["value"]} for item in message]


def run_qwen2(mode: str, image_path: str, dataset: str, template_name: str, template_on_last: bool, limit_mm_per_prompt: int) -> dict[str, Any]:
    probe = Qwen2Probe(mode, template_name, template_on_last, limit_mm_per_prompt)
    inputs = build_test_input(image_path)
    return {
        "prepare_content": summarize_content(probe._prepare_content(inputs, dataset=dataset)),
        "prepare_content_vllm": summarize_content(probe._prepare_content_vllm(inputs, dataset=dataset)),
    }


def run_qwen35(mode: str, image_path: str, dataset: str, template_name: str, template_on_last: bool, limit_mm_per_prompt: int) -> dict[str, Any]:
    probe = Qwen35Probe(mode, template_name, template_on_last, limit_mm_per_prompt)
    inputs = build_test_input(image_path)
    result = {
        "prepare_content": summarize_content(probe._prepare_content(inputs, dataset=dataset)),
    }
    if hasattr(probe, "_prepare_content_vllm"):
        result["prepare_content_vllm"] = summarize_content(probe._prepare_content_vllm(inputs, dataset=dataset))
    return result


def run_minicpm26(mode: str, image_path: str, template_name: str, template_on_last: bool) -> dict[str, Any]:
    probe = MiniCPM26Probe(mode, template_name, template_on_last)
    return {"message": summarize_content(probe._apply_replay_pipeline(prepare_minicpm_message(build_test_input(image_path))))}


def run_minicpm45(mode: str, image_path: str, dataset: str, template_name: str, template_on_last: bool) -> dict[str, Any]:
    probe = MiniCPM45Probe(mode, template_name, template_on_last)
    return {"message": summarize_content(probe._apply_replay_pipeline(prepare_minicpm_message(build_test_input(image_path)), dataset=dataset))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe replay input construction across representative model wrappers without loading model weights.")
    parser.add_argument("--image-path", type=str, default="image_test.png")
    parser.add_argument("--dataset", type=str, default="OCRBench")
    parser.add_argument("--limit-mm-per-prompt", type=int, default=8)
    parser.add_argument("--template-on-last-replay-text", action="store_true")
    parser.add_argument("--template-names", nargs="*", default=[PROMPT_TEMPLATE_IDENTITY, PROMPT_TEMPLATE_DIRECTLY_ANSWER])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = os.path.abspath(args.image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path not found: {image_path}")

    modes = [
        REPLAY_IMAGE_TEXT,
        REPLAY_TEXT_IMAGE,
        REPLAY_IMAGE_TEXT_TEXT,
        REPLAY_IMAGE_TEXT_IMAGE,
        REPLAY_IMAGE_TEXT_IMAGE_TEXT,
        REPLAY_IMAGE_IMAGE_TEXT,
    ]
    results = []
    for template_name in args.template_names:
        for mode in modes:
            results.append({
                "template_name": template_name,
                "mode": mode,
                "qwen2": run_qwen2(mode, image_path, args.dataset, template_name, args.template_on_last_replay_text, args.limit_mm_per_prompt),
                "qwen35": run_qwen35(mode, image_path, args.dataset, template_name, args.template_on_last_replay_text, args.limit_mm_per_prompt),
                "minicpm26": run_minicpm26(mode, image_path, template_name, args.template_on_last_replay_text),
                "minicpm45": run_minicpm45(mode, image_path, args.dataset, template_name, args.template_on_last_replay_text),
            })

    print(json.dumps({
        "image_path": image_path,
        "dataset": args.dataset,
        "template_on_last_replay_text": args.template_on_last_replay_text,
        "results": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
