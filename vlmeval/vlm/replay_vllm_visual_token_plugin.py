from __future__ import annotations

import os

from .replay_visual_token_shift import canonicalize_visual_token_shift


def register() -> None:
    mode = canonicalize_visual_token_shift(os.environ.get("REPLAY_VISUAL_TOKEN_SHIFT", "none"))
    if mode == "none":
        return

    from vllm import ModelRegistry

    target_family = os.environ.get("REPLAY_VLLM_TARGET_FAMILY", "").strip()
    if target_family == "qwen2_5_vl":
        ModelRegistry.register_model(
            "Qwen2_5_VLForConditionalGeneration",
            "vlmeval.vlm.replay_vllm_visual_token_models:ReplayShiftQwen2_5VL",
        )
        return
    if target_family == "minicpm_o_4_5":
        ModelRegistry.register_model(
            "MiniCPMO",
            "vlmeval.vlm.replay_vllm_visual_token_models:ReplayShiftMiniCPMO45",
        )
        return
    raise RuntimeError(
        "REPLAY_VLLM_TARGET_FAMILY must be qwen2_5_vl or minicpm_o_4_5 "
        f"when visual-token shift is enabled, got {target_family!r}"
    )
