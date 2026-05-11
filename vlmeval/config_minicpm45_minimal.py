import os
from functools import partial

from vlmeval.vlm import MiniCPM_V_4_5, MiniCPM_V_4_5_Replay


def _env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


model_path = os.getenv("MODEL_PATH", "default")
model_root = os.getenv("MODEL_ROOT", "/models")
model_path_minicpm45 = os.getenv("MODEL_PATH_MINICPM45") or f"{model_root}/MiniCPM-V-4_5"
model_path_minicpmo45 = os.getenv("MODEL_PATH_MINICPMO45") or f"{model_root}/MiniCPM-o-4_5"
use_vllm = _env_bool("MINICPM45_USE_VLLM", True)


supported_VLM = {
    "MiniCPM-V-4_5": partial(
        MiniCPM_V_4_5,
        model_path=model_path if model_path != "default" else model_path_minicpm45,
        use_vllm=use_vllm,
    ),
    "MiniCPM-V-4_5-Replay": partial(
        MiniCPM_V_4_5_Replay,
        model_path=model_path if model_path != "default" else model_path_minicpm45,
        use_vllm=use_vllm,
    ),
    "MiniCPM-o-4_5": partial(
        MiniCPM_V_4_5,
        model_path=model_path if model_path != "default" else model_path_minicpmo45,
        use_vllm=use_vllm,
    ),
    "MiniCPM-o-4_5-Replay": partial(
        MiniCPM_V_4_5_Replay,
        model_path=model_path if model_path != "default" else model_path_minicpmo45,
        use_vllm=use_vllm,
    ),
}
