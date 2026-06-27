import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# Temporarily bypass SSL certificate verification to download files from oss.

try:
    import torch
except ImportError:
    pass

from .smp import *
load_env()


def _minimal_runtime_registry_enabled():
    flags = (
        'VLMEVAL_USE_QWEN_MINIMAL_CONFIG',
        'VLMEVAL_USE_API_REPLAY_MINIMAL_CONFIG',
        'VLMEVAL_USE_MINICPM45_MINIMAL_CONFIG',
        'VLMEVAL_USE_GEMMA3_MINIMAL_CONFIG',
    )
    truthy = {'1', 'true', 'yes', 'on'}
    return any(str(os.environ.get(flag, '0')).strip().lower() in truthy for flag in flags)


if os.environ.get('VLMEVAL_LAZY_INIT', '0').strip().lower() not in {'1', 'true', 'yes', 'on'}:
    from .api import *
    from .dataset import *
    from .utils import *
    from .vlm import *

    if not _minimal_runtime_registry_enabled():
        from .config import *
        from .tools import cli
    else:
        cli = None
else:
    cli = None


__version__ = '0.2rc1'
