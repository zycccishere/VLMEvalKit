import os


def _truthy(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


if _truthy("VLMEVAL_USE_QWEN_MINIMAL_CONFIG"):
    from .config_qwen_minimal import supported_VLM
elif _truthy("VLMEVAL_USE_API_REPLAY_MINIMAL_CONFIG"):
    from .config_api_replay_minimal import supported_VLM
elif _truthy("VLMEVAL_USE_MINICPM45_MINIMAL_CONFIG"):
    from .config_minicpm45_minimal import supported_VLM
elif _truthy("VLMEVAL_USE_GEMMA3_MINIMAL_CONFIG"):
    from .config_gemma3_minimal import supported_VLM
elif _truthy("VLMEVAL_LAZY_INIT") or _truthy("VLMEVAL_VLM_MINIMAL_IMPORT"):
    supported_VLM = {}
else:
    from .config import supported_VLM
