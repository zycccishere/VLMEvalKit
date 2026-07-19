#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def make_smoke_image(path: Path) -> None:
    height = width = 448
    y, x = np.mgrid[0:height, 0:width]
    pixels = np.stack(
        [
            (x * 255 // (width - 1)),
            (y * 255 // (height - 1)),
            ((x // 28 + y // 28) % 2) * 255,
        ],
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def configure(dump_dir: Path) -> None:
    values = {
        "REPLAY_MODE": "image_text_image_text",
        "REPLAY_TIMES": "1",
        "REPLAY_IMAGE_COPY_MODE": "reuse_path",
        "REPLAY_PROMPT_TEMPLATE_NAME": "identity",
        "REPLAY_TEMPLATE_ON_LAST_REPLAY_TEXT": "1",
        "REPLAY_IMAGE_TRANSFORM": "baseline",
        "REPLAY_VISUAL_TOKEN_SHIFT": "roll_right_1",
        "REPLAY_VISUAL_TOKEN_SHIFT_TARGET_POSITION": "2",
        "REPLAY_VISUAL_TOKEN_SHIFT_STRICT": "1",
        "REPLAY_VISUAL_TOKEN_SHIFT_RAW_DUMP": "1",
        "REPLAY_VISUAL_TOKEN_SHIFT_DUMP_SAMPLES": "1",
        "REPLAY_VISUAL_TOKEN_SHIFT_DUMP_DIR": str(dump_dir),
        "REPLAY_SAFE_FALLBACK": "0",
        "REPLAY_TRACE_LEVEL": "summary",
        "REPLAY_TRACE_SAMPLES": "1",
        "REPLAY_TRACE_DIR": str(dump_dir),
        "MINICPM45_USE_VLLM": "0",
        "MINICPM45_MAX_NEW_TOKENS": "1",
        "VLMEVAL_API_MINIMAL_IMPORT": "1",
        "VLMEVAL_VLM_MINIMAL_IMPORT": "1",
        "VLMEVAL_LAZY_INIT": "1",
    }
    os.environ.update(values)


def load_model(family: str, model_path: str):
    if family == "qwen":
        from vlmeval.vlm.qwen2_vl.model import Qwen2VLChatReplay

        return Qwen2VLChatReplay(
            model_path=model_path,
            min_pixels=1280 * 28 * 28,
            max_pixels=16384 * 28 * 28,
            use_custom_prompt=False,
            use_vllm=False,
            max_new_tokens=1,
            top_p=0.001,
            top_k=1,
            temperature=0.01,
        )
    if family == "minicpm":
        from vlmeval.vlm.minicpm_v_4_5_replay import MiniCPM_V_4_5_Replay

        return MiniCPM_V_4_5_Replay(
            model_path=model_path,
            use_vllm=False,
            max_new_tokens=1,
            sampling=False,
            num_beams=1,
        )
    raise ValueError(f"unsupported family: {family}")


def validate_dump(dump_dir: Path, family: str, model_path: Path, output: str) -> dict:
    records = []
    for path in sorted(dump_dir.glob("visual_token_shift.pid*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(records) != 1:
        raise AssertionError(f"expected exactly one real visual-token record, found {len(records)}")
    record = records[0]
    raw_path = Path(record["raw_npz_path"])
    with np.load(raw_path) as raw:
        before = raw["target_before"]
        after = raw["target_after"]
        mapping = raw["source_index_for_output"]
        non_target_before = raw["non_target_before"]
        non_target_after = raw["non_target_after"]
        raw_checks = {
            "target_exact_roll": bool(np.array_equal(after, before[:, mapping, :])),
            "target_expected_mapping": bool(
                np.array_equal(mapping, np.asarray([len(mapping) - 1] + list(range(len(mapping) - 1))))
            ),
            "non_target_bitwise_unchanged": bool(np.array_equal(non_target_before, non_target_after)),
        }

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    checks = {
        **raw_checks,
        "real_model_path": Path(record["model_name"]).resolve() == model_path.resolve(),
        "iqiq_topology_validated": bool(record["iqiq_topology"].get("validated")),
        "original_one_image": record["iqiq_topology"].get("original_image_count") == 1,
        "replayed_two_images": record["iqiq_topology"].get("replayed_image_count") == 2,
        "i1_i2_same_reference": bool(record["iqiq_topology"].get("i1_i2_same_reference")),
        "q1_q2_same_text": bool(record["iqiq_topology"].get("q1_q2_same_text")),
        "shape_unchanged": record["input_shape"] == record["output_shape"],
        "dtype_unchanged": bool(record["dtype_unchanged"]),
        "roll_exact_in_hook": bool(record["exact_roll_verified"]) and record["max_abs_error"] == 0.0,
        "final_container_exact": bool(record["final_target_exact"]) and bool(record["final_non_target_exact"]),
        "hook_applied_once": record["apply_count"] == 1,
        "target_is_i2": record["target_image_position"] == 2,
        "generated_once": isinstance(output, str),
    }
    if family == "qwen":
        checks.update(
            {
                "qwen_architecture": "Qwen2_5_VLForConditionalGeneration" in config.get("architectures", []),
                "qwen_two_equal_image_spans": record["image_count"] == 2
                and record["image_token_counts"][0] == record["image_token_counts"][1],
                "qwen_identical_grid": record["qwen_grid_thw"][0] == record["qwen_grid_thw"][1],
                "qwen_post_merger_stage": record["stage"] == "qwen_post_spatial_merger_pre_llm_injection",
                "qwen_transformers_5_output_abi": record["qwen_visual_output_kind"]
                == "base_model_output_with_pooling",
            }
        )
    else:
        checks.update(
            {
                "minicpm_architecture": "MiniCPMO" in config.get("architectures", []),
                "minicpm_query_num_64": record["minicpm_query_num"] == 64
                and record["token_count_per_block"] == 64,
                "minicpm_two_payload_bounds": record["image_count"] == 2
                and record["image_placeholder_lengths"] == [64, 64],
                "minicpm_pre_pop_payload_captured": record["minicpm_processor_payload"]["image_count"] == 2
                and record["minicpm_processor_payload"]["pixel_block_count"] == 2
                and record["minicpm_processor_payload"]["capture_count"] == 1
                and record["minicpm_processor_payload"]["image_bound_lengths"] == [64, 64],
                "minicpm_post_resampler_stage": record["stage"] == "minicpm_post_resampler_pre_llm_injection",
            }
        )

    return {
        "all_passed": all(checks.values()),
        "family": family,
        "model_path": str(model_path),
        "architectures": config.get("architectures", []),
        "output_preview": output[:200],
        "record": record,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["qwen", "minicpm"], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path, required=True)
    args = parser.parse_args()

    args.dump_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.dump_dir / "smoke_input.png"
    make_smoke_image(image_path)
    configure(args.dump_dir)

    model = load_model(args.family, str(args.model_path))
    message = [
        {
            "type": "image",
            "value": str(image_path),
            "replay_meta": {"sample_index": "visual-token-shift-real-smoke"},
        },
        {"type": "text", "value": "Describe the image in one word."},
    ]
    output = model.generate_inner(message, dataset=None)
    summary = validate_dump(args.dump_dir, args.family, args.model_path, output)
    summary_path = args.dump_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
