import os
from functools import partial

os.environ.setdefault("VLMEVAL_VLM_MINIMAL_IMPORT", "1")

from vlmeval.vlm import Gemma4Replay


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


model_path = os.getenv("MODEL_PATH", "default")
model_root = os.getenv("MODEL_ROOT", "/models")
model_path_gemma4_e2b = os.getenv("MODEL_PATH_GEMMA4_E2B") or f"{model_root}/gemma-4-E2B-it"
model_path_gemma4_e4b = os.getenv("MODEL_PATH_GEMMA4_E4B") or f"{model_root}/gemma-4-E4B-it"
model_path_gemma4_26b_a4b = os.getenv("MODEL_PATH_GEMMA4_26B_A4B") or f"{model_root}/gemma-4-26B-A4B-it"
model_path_gemma4_31b = os.getenv("MODEL_PATH_GEMMA4_31B") or f"{model_root}/gemma-4-31B-it"
use_vllm = _env_bool("GEMMA4_USE_VLLM", True)
max_new_tokens = int(os.getenv("GEMMA4_MAX_NEW_TOKENS", "4096"))


supported_VLM = {
    "Gemma4Replay": partial(
        Gemma4Replay,
        model_path=model_path,
        use_custom_prompt=True,
        use_vllm=use_vllm,
        max_new_tokens=max_new_tokens,
    ),
    "Gemma4-E2B-it-Replay": partial(
        Gemma4Replay,
        model_path=model_path if model_path != "default" else model_path_gemma4_e2b,
        use_custom_prompt=True,
        use_vllm=use_vllm,
        max_new_tokens=max_new_tokens,
    ),
    "Gemma4-E4B-it-Replay": partial(
        Gemma4Replay,
        model_path=model_path if model_path != "default" else model_path_gemma4_e4b,
        use_custom_prompt=True,
        use_vllm=use_vllm,
        max_new_tokens=max_new_tokens,
    ),
    "Gemma4-26B-A4B-it-Replay": partial(
        Gemma4Replay,
        model_path=model_path if model_path != "default" else model_path_gemma4_26b_a4b,
        use_custom_prompt=True,
        use_vllm=use_vllm,
        max_new_tokens=max_new_tokens,
    ),
    "Gemma4-31B-it-Replay": partial(
        Gemma4Replay,
        model_path=model_path if model_path != "default" else model_path_gemma4_31b,
        use_custom_prompt=True,
        use_vllm=use_vllm,
        max_new_tokens=max_new_tokens,
    ),
}
