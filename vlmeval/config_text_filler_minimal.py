import os
from functools import partial

os.environ.setdefault("VLMEVAL_VLM_MINIMAL_IMPORT", "1")

from vlmeval.vlm import Gemma3Replay, MiniCPM_V_4_5, MiniCPM_V_4_5_Replay


def _env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


model_path = os.getenv("MODEL_PATH", "default")
model_root = os.getenv("MODEL_ROOT", "/models")
model_path_gemma3_12b = os.getenv("MODEL_PATH_GEMMA3_12B") or f"{model_root}/gemma-3-12b-it"
model_path_minicpmo45 = os.getenv("MODEL_PATH_MINICPMO45") or f"{model_root}/MiniCPM-o-4_5"
gemma3_use_vllm = _env_bool("GEMMA3_USE_VLLM", True)
gemma3_max_new_tokens = int(os.getenv("GEMMA3_MAX_NEW_TOKENS", "4096"))
minicpm_use_vllm = _env_bool("MINICPM45_USE_VLLM", True)


supported_VLM = {
    "Gemma3-12B": partial(
        Gemma3Replay,
        model_path=model_path if model_path != "default" else model_path_gemma3_12b,
        use_custom_prompt=True,
        use_vllm=gemma3_use_vllm,
        max_new_tokens=gemma3_max_new_tokens,
    ),
    "Gemma3-12B-Replay": partial(
        Gemma3Replay,
        model_path=model_path if model_path != "default" else model_path_gemma3_12b,
        use_custom_prompt=True,
        use_vllm=gemma3_use_vllm,
        max_new_tokens=gemma3_max_new_tokens,
    ),
    "MiniCPM-o-4_5": partial(
        MiniCPM_V_4_5,
        model_path=model_path if model_path != "default" else model_path_minicpmo45,
        use_vllm=minicpm_use_vllm,
    ),
    "MiniCPM-o-4_5-Replay": partial(
        MiniCPM_V_4_5_Replay,
        model_path=model_path if model_path != "default" else model_path_minicpmo45,
        use_vllm=minicpm_use_vllm,
    ),
}
