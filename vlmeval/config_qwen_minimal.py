import os
from functools import partial

os.environ.setdefault("VLMEVAL_VLM_MINIMAL_IMPORT", "1")

from vlmeval.vlm.qwen2_vl.model import Qwen2VLChatReplay


model_path = os.getenv("MODEL_PATH", "default")
model_root = os.getenv("MODEL_ROOT", "/models")
model_path_qwen25_3b = os.getenv("MODEL_PATH_QWEN25_3B") or f"{model_root}/Qwen2.5-VL-3B-Instruct"
model_path_qwen25_7b = os.getenv("MODEL_PATH_QWEN25") or f"{model_root}/Qwen2.5-VL-7B-Instruct"
model_path_qwen25_32b = os.getenv("MODEL_PATH_QWEN25_32B") or f"{model_root}/Qwen2.5-VL-32B-Instruct"
model_path_qwen25_72b = os.getenv("MODEL_PATH_QWEN25_72B") or f"{model_root}/Qwen2.5-VL-72B-Instruct"


def _qwen25_replay(model_path_value: str):
    return partial(
        Qwen2VLChatReplay,
        model_path=model_path_value,
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        use_vllm=True,
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
}
