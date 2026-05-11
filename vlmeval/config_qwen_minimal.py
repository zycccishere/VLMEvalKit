import os
from functools import partial

os.environ.setdefault("VLMEVAL_VLM_MINIMAL_IMPORT", "1")

from vlmeval.vlm import Qwen3VLChat, Qwen35VLChatReplay
from vlmeval.vlm.qwen2_vl.model import Qwen2VLChatReplay


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, str(default)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


model_path = os.getenv("MODEL_PATH", "default")
model_root = os.getenv("MODEL_ROOT", "/models")
model_path_qwen2 = os.getenv("MODEL_PATH_QWEN2") or f"{model_root}/Qwen2-VL-7B-Instruct"
model_path_qwen25 = os.getenv("MODEL_PATH_QWEN25") or f"{model_root}/Qwen2.5-VL-7B-Instruct"
model_path_qwen35_4b = os.getenv("MODEL_PATH_QWEN35_4B") or f"{model_root}/Qwen3.5-4B"
model_path_qwen35_9b = os.getenv("MODEL_PATH_QWEN35_9B") or f"{model_root}/Qwen3.5-9B"
model_path_qwen35_27b = os.getenv("MODEL_PATH_QWEN35_27B") or f"{model_root}/Qwen3.5-27B"
model_path_qwen35_35b_a3b = os.getenv("MODEL_PATH_QWEN35_35B_A3B") or f"{model_root}/Qwen3.5-35B-A3B"
use_vllm = _env_bool("QWEN35_USE_VLLM", False)


supported_VLM = {
    "Qwen2VLChatReplay": partial(
        Qwen2VLChatReplay,
        model_path=model_path,
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2-VL-7B-Instruct-Replay": partial(
        Qwen2VLChatReplay,
        model_path=model_path_qwen2,
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-7B-Instruct-Replay": partial(
        Qwen2VLChatReplay,
        model_path=model_path_qwen25,
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-3B-Instruct-Replay": partial(
        Qwen2VLChatReplay,
        model_path=model_path,
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen3VLChat": partial(
        Qwen3VLChat,
        model_path=model_path,
        use_custom_prompt=False,
        use_vllm=use_vllm,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        max_new_tokens=32768,
    ),
    "Qwen35VLChatReplay": partial(
        Qwen35VLChatReplay,
        model_path=model_path,
        use_custom_prompt=False,
        use_vllm=use_vllm,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        max_new_tokens=32768,
    ),
    "Qwen3.5-4B": partial(
        Qwen3VLChat,
        model_path=model_path_qwen35_4b,
        use_custom_prompt=False,
        use_vllm=use_vllm,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        max_new_tokens=32768,
    ),
    "Qwen3.5-9B": partial(
        Qwen3VLChat,
        model_path=model_path_qwen35_9b,
        use_custom_prompt=False,
        use_vllm=use_vllm,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        max_new_tokens=32768,
    ),
    "Qwen3.5-27B": partial(
        Qwen3VLChat,
        model_path=model_path_qwen35_27b,
        use_custom_prompt=False,
        use_vllm=use_vllm,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        max_new_tokens=32768,
    ),
    "Qwen3.5-35B-A3B": partial(
        Qwen3VLChat,
        model_path=model_path_qwen35_35b_a3b,
        use_custom_prompt=False,
        use_vllm=use_vllm,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        max_new_tokens=32768,
    ),
    "Qwen3.5-4B-Replay": partial(
        Qwen35VLChatReplay,
        model_path=model_path_qwen35_4b,
        use_custom_prompt=False,
        use_vllm=use_vllm,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        max_new_tokens=32768,
    ),
    "Qwen3.5-9B-Replay": partial(
        Qwen35VLChatReplay,
        model_path=model_path_qwen35_9b,
        use_custom_prompt=False,
        use_vllm=use_vllm,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        max_new_tokens=32768,
    ),
    "Qwen3.5-27B-Replay": partial(
        Qwen35VLChatReplay,
        model_path=model_path_qwen35_27b,
        use_custom_prompt=False,
        use_vllm=use_vllm,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        max_new_tokens=32768,
    ),
    "Qwen3.5-35B-A3B-Replay": partial(
        Qwen35VLChatReplay,
        model_path=model_path_qwen35_35b_a3b,
        use_custom_prompt=False,
        use_vllm=use_vllm,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        max_new_tokens=32768,
    ),
}
