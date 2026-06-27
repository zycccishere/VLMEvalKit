import os
from functools import partial

os.environ.setdefault("VLMEVAL_API_MINIMAL_IMPORT", "1")
os.environ.setdefault("VLMEVAL_VLM_MINIMAL_IMPORT", "1")

from vlmeval.api import GPT4VReplay

API_MAX_TOKENS = int(os.environ.get("VLMEVAL_API_MAX_TOKENS", "2048"))
API_TIMEOUT = int(os.environ.get("VLMEVAL_API_TIMEOUT", "60"))
API_IMG_SIZE = int(os.environ.get("VLMEVAL_API_IMG_SIZE", "-1"))


supported_VLM = {
    "gpt-4o-mini": partial(
        GPT4VReplay,
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=API_MAX_TOKENS,
        timeout=API_TIMEOUT,
        img_size=API_IMG_SIZE,
        img_detail="high",
        retry=10,
    ),
    "gpt-5-mini": partial(
        GPT4VReplay,
        model="gpt-5-mini",
        temperature=0,
        max_tokens=API_MAX_TOKENS,
        timeout=API_TIMEOUT,
        img_size=API_IMG_SIZE,
        img_detail="high",
        retry=10,
        reasoning_effort="minimal",
    ),
    "gpt-5-2025-08-07": partial(
        GPT4VReplay,
        model="gpt-5-2025-08-07",
        temperature=0,
        max_tokens=API_MAX_TOKENS,
        timeout=API_TIMEOUT,
        img_size=API_IMG_SIZE,
        img_detail="high",
        retry=10,
        reasoning_effort="minimal",
    ),
    "gpt-5-chat": partial(
        GPT4VReplay,
        model="gpt-5-chat",
        temperature=0,
        max_tokens=API_MAX_TOKENS,
        timeout=API_TIMEOUT,
        img_size=API_IMG_SIZE,
        img_detail="high",
        retry=10,
    ),
    "claude-haiku-4-5-20251001": partial(
        GPT4VReplay,
        model="claude-haiku-4-5-20251001",
        temperature=0,
        max_tokens=API_MAX_TOKENS,
        timeout=API_TIMEOUT,
        img_size=API_IMG_SIZE,
        img_detail="high",
        retry=10,
    ),
    "gemini-2.5-flash-lite": partial(
        GPT4VReplay,
        model="gemini-2.5-flash-lite",
        temperature=0,
        max_tokens=API_MAX_TOKENS,
        timeout=API_TIMEOUT,
        img_size=API_IMG_SIZE,
        img_detail="high",
        retry=10,
    ),
    "gemini-2.5-flash-nothinking": partial(
        GPT4VReplay,
        model="gemini-2.5-flash-nothinking",
        temperature=0,
        max_tokens=API_MAX_TOKENS,
        timeout=API_TIMEOUT,
        img_size=API_IMG_SIZE,
        img_detail="high",
        retry=10,
    ),
    "gemini-2.5-flash-thinking": partial(
        GPT4VReplay,
        model="gemini-2.5-flash-thinking",
        temperature=0,
        max_tokens=API_MAX_TOKENS,
        timeout=API_TIMEOUT,
        img_size=API_IMG_SIZE,
        img_detail="high",
        retry=10,
    ),
    "gemini-3-flash-preview-nothinking": partial(
        GPT4VReplay,
        model="gemini-3-flash-preview-nothinking",
        temperature=0,
        max_tokens=API_MAX_TOKENS,
        timeout=API_TIMEOUT,
        img_size=API_IMG_SIZE,
        img_detail="high",
        retry=10,
    ),
    "gemini-3.1-flash-lite": partial(
        GPT4VReplay,
        model="gemini-3.1-flash-lite",
        temperature=0,
        max_tokens=API_MAX_TOKENS,
        timeout=API_TIMEOUT,
        img_size=API_IMG_SIZE,
        img_detail="high",
        retry=10,
    ),
}
