import os


if os.environ.get("VLMEVAL_USE_QWEN_MINIMAL_CONFIG", "0").strip().lower() in {"1", "true", "yes", "on"}:
    from .config_qwen_minimal import supported_VLM
elif os.environ.get("VLMEVAL_USE_API_REPLAY_MINIMAL_CONFIG", "0").strip().lower() in {"1", "true", "yes", "on"}:
    from .config_api_replay_minimal import supported_VLM
elif os.environ.get("VLMEVAL_USE_MINICPM45_MINIMAL_CONFIG", "0").strip().lower() in {"1", "true", "yes", "on"}:
    from .config_minicpm45_minimal import supported_VLM
elif os.environ.get("VLMEVAL_USE_GEMMA3_MINIMAL_CONFIG", "0").strip().lower() in {"1", "true", "yes", "on"}:
    from .config_gemma3_minimal import supported_VLM
elif os.environ.get("VLMEVAL_USE_GEMMA4_MINIMAL_CONFIG", "0").strip().lower() in {"1", "true", "yes", "on"}:
    from .config_gemma4_minimal import supported_VLM
else:
    from .config import supported_VLM
