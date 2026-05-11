import os
from functools import partial

os.environ.setdefault("VLMEVAL_VLM_MINIMAL_IMPORT", "1")

from vlmeval.vlm import Gemma3Replay


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


model_path = os.getenv("MODEL_PATH", "default")
model_root = os.getenv("MODEL_ROOT", "/models")
model_path_gemma3_4b = os.getenv("MODEL_PATH_GEMMA3_4B") or f"{model_root}/gemma-3-4b-it"
model_path_gemma3_12b = os.getenv("MODEL_PATH_GEMMA3_12B") or f"{model_root}/gemma-3-12b-it"
model_path_gemma3_27b = os.getenv("MODEL_PATH_GEMMA3_27B") or f"{model_root}/gemma-3-27b-it"
use_vllm = _env_bool("GEMMA3_USE_VLLM", True)
max_new_tokens = int(os.getenv("GEMMA3_MAX_NEW_TOKENS", "4096"))


supported_VLM = {
    "Gemma3Replay": partial(
        Gemma3Replay,
        model_path=model_path,
        use_custom_prompt=True,
        use_vllm=use_vllm,
        max_new_tokens=max_new_tokens,
    ),
    "Gemma3-4B": partial(
        Gemma3Replay,
        model_path=model_path if model_path != "default" else model_path_gemma3_4b,
        use_custom_prompt=True,
        use_vllm=use_vllm,
        max_new_tokens=max_new_tokens,
    ),
    "Gemma3-12B": partial(
        Gemma3Replay,
        model_path=model_path if model_path != "default" else model_path_gemma3_12b,
        use_custom_prompt=True,
        use_vllm=use_vllm,
        max_new_tokens=max_new_tokens,
    ),
    "Gemma3-27B": partial(
        Gemma3Replay,
        model_path=model_path if model_path != "default" else model_path_gemma3_27b,
        use_custom_prompt=True,
        use_vllm=use_vllm,
        max_new_tokens=max_new_tokens,
    ),
    "Gemma3-4B-Replay": partial(
        Gemma3Replay,
        model_path=model_path if model_path != "default" else model_path_gemma3_4b,
        use_custom_prompt=True,
        use_vllm=use_vllm,
        max_new_tokens=max_new_tokens,
    ),
    "Gemma3-12B-Replay": partial(
        Gemma3Replay,
        model_path=model_path if model_path != "default" else model_path_gemma3_12b,
        use_custom_prompt=True,
        use_vllm=use_vllm,
        max_new_tokens=max_new_tokens,
    ),
    "Gemma3-27B-Replay": partial(
        Gemma3Replay,
        model_path=model_path if model_path != "default" else model_path_gemma3_27b,
        use_custom_prompt=True,
        use_vllm=use_vllm,
        max_new_tokens=max_new_tokens,
    ),
}
