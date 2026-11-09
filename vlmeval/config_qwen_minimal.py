import os
from functools import partial

os.environ.setdefault("VLMEVAL_VLM_MINIMAL_IMPORT", "1")

from vlmeval.vlm.qwen2_vl.model import Qwen2VLChatReplay
from vlmeval.vlm.qwen35_vl_replay import Qwen35VLChatReplay


model_path = os.getenv("MODEL_PATH", "default")
model_root = os.getenv("MODEL_ROOT", "/models")
model_path_qwen25_3b = os.getenv("MODEL_PATH_QWEN25_3B") or f"{model_root}/Qwen2.5-VL-3B-Instruct"
model_path_qwen25_7b = os.getenv("MODEL_PATH_QWEN25") or f"{model_root}/Qwen2.5-VL-7B-Instruct"
model_path_qwen25_32b = os.getenv("MODEL_PATH_QWEN25_32B") or f"{model_root}/Qwen2.5-VL-32B-Instruct"
model_path_qwen25_72b = os.getenv("MODEL_PATH_QWEN25_72B") or f"{model_root}/Qwen2.5-VL-72B-Instruct"
model_path_qwen35_moe_35b = os.getenv("MODEL_PATH_QWEN35MOE_35B") or f"{model_root}/Qwen3.5-35B-A3B"
model_path_qwen38_27b = os.getenv("MODEL_PATH_QWEN38_27B") or "/cloud/oss_checkpoints/Qwen3/Qwen3.8-27B"


def _qwen25_replay(model_path_value: str):
    return partial(
        Qwen2VLChatReplay,
        model_path=model_path_value,
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        use_vllm=True,
    )


def _qwen38_qwen35moe_replay(model_path_value: str, thinking: str, presence_penalty: float):
    # Sampling parameters follow the model cards; see the matching entries in
    # vlmeval/config.py for the documented rationale.
    return partial(
        Qwen35VLChatReplay,
        model_path=model_path_value,
        use_custom_prompt=False,
        use_vllm=True,
        thinking=thinking,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        presence_penalty=presence_penalty,
        repetition_penalty=1.0,
        max_new_tokens=32768,
    )


# Qwen2VLChatReplay is the wrapper class name reused by the Qwen2.5-VL replay
# route. It is kept as a generic alias for matrix entries that pass MODEL_PATH;
# the active release model aliases are the four Qwen2.5-VL sizes below.
supported_VLM = {
    "Qwen2VLChatReplay": _qwen25_replay(model_path),
    "Qwen2.5-VL-3B-Instruct-Replay": _qwen25_replay(model_path_qwen25_3b),
    "Qwen2.5-VL-7B-Instruct-Replay": _qwen25_replay(model_path_qwen25_7b),
    "Qwen2.5-VL-32B-Instruct-Replay": _qwen25_replay(model_path_qwen25_32b),
    "Qwen2.5-VL-72B-Instruct-Replay": _qwen25_replay(model_path_qwen25_72b),
    # Qwen3.8-27B / Qwen3.5-35B-A3B replay entries with explicit thinking
    # control; `-Think-` keeps the chat template's thinking mode on, plain
    # entries disable it (non-thinking sampling profile in the wrapper).
    # presence_penalty: 0.0 for Qwen3.8-27B thinking (its model card), 1.5 for
    # Qwen3.5-35B-A3B (its model card).
    "Qwen3.8-27B-Replay": _qwen38_qwen35moe_replay(model_path_qwen38_27b, thinking="off", presence_penalty=0.0),
    "Qwen3.8-27B-Think-Replay": _qwen38_qwen35moe_replay(model_path_qwen38_27b, thinking="on", presence_penalty=0.0),
    "Qwen3.5-35B-A3B-Replay": _qwen38_qwen35moe_replay(model_path_qwen35_moe_35b, thinking="off", presence_penalty=1.5),
    "Qwen3.5-35B-A3B-Think-Replay": _qwen38_qwen35moe_replay(model_path_qwen35_moe_35b, thinking="on", presence_penalty=1.5),
}
