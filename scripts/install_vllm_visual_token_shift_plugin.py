#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DIST_INFO = "vlmeval_vllm_visual_token_shift_plugin-0.1.0.dist-info"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()

    dist_info = args.target / DIST_INFO
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        "Name: vlmeval-vllm-visual-token-shift-plugin\n"
        "Version: 0.1.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[vllm.general_plugins]\n"
        "vlmeval_visual_token_shift = "
        "vlmeval.vlm.replay_vllm_visual_token_plugin:register\n",
        encoding="utf-8",
    )
    (dist_info / "top_level.txt").write_text("vlmeval\n", encoding="utf-8")
    print(dist_info)


if __name__ == "__main__":
    main()
