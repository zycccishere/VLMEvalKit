#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test LLaVA replay modes.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--question", default="What is in the image? Answer briefly.")
    parser.add_argument("--modes", default="image_text,text_image,image_text_image")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if args.backend == "vllm":
        os.environ["LLAVA_USE_VLLM"] = "1"

    from vlmeval.vlm.llava import LLaVA_HF_Replay

    msg = [
        {"type": "image", "value": args.image_path},
        {"type": "text", "value": args.question},
    ]
    model = LLaVA_HF_Replay(model_path=args.model_path, max_new_tokens=args.max_new_tokens)

    results: dict[str, dict] = {}
    for mode in [x.strip() for x in args.modes.split(",") if x.strip()]:
        model.replay_cfg = {
            "mode": mode,
            "repeat_times": 1,
            "debug": False,
            "image_copy_mode": "reuse_path",
        }
        prepared = model._prepare_content(msg)
        prompt, _ = model._build_prompt_and_images(msg)
        output = model.generate_inner(msg)
        results[mode] = {
            "prepared": prepared,
            "prompt": prompt,
            "output": output,
        }

    payload = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
