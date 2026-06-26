import os

from .prompt import Qwen2VLPromptMixin

if os.environ.get('VLMEVAL_VLM_MINIMAL_IMPORT', '0').strip().lower() not in {'1', 'true', 'yes', 'on'}:
    from .model import Qwen2VLChat, Qwen2VLChatAguvis, Qwen2VLChatReplay
