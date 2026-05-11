#!/usr/bin/env python3
import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_BASE = "https://api.openai.com/v1"
# 当你想让claude模型为你完成主线任务/代码生成时，使用这个模型
DEFAULT_MODEL_FOR_CALLING = "gpt-4o-mini"
# 当你想在代码中对benchmarks进行评估时，使用这个模型
DEFAULT_MODEL_FOR_EVALUATING = "gpt-4o-mini"
DEFAULT_API_KEY = ""

# 支持的模型列表
# Claude:
# - claude-opus-4-6-thinking
# OpenAI:
# - gpt-4o
# - gpt-5.4
# Gemini:
# - gemini-3.1-pro-preview-thinking
# - gemini-3-pro-image-preview


def resolve_api_key(cli_value: str | None) -> str:
    if cli_value:
        return cli_value.strip()
    return (
        os.environ.get("OPENAI_API_KEY_JUDGE", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("OPENAI_COMPATIBLE_API_KEY", "").strip()
        or DEFAULT_API_KEY
    )


def resolve_api_base(cli_value: str | None) -> str:
    base = (
        cli_value
        or os.environ.get("OPENAI_API_BASE", "").strip()
        or os.environ.get("OPENAI_API_BASE_JUDGE", "").strip()
        or os.environ.get("OPENAI_COMPATIBLE_API_BASE", "").strip()
        or DEFAULT_BASE
    )
    normalized = base.rstrip("/")
    lower = normalized.lower()
    if lower.endswith("/chat/completions"):
        return normalized
    if lower.endswith("/v1"):
        return normalized + "/chat/completions"
    if lower.endswith("/v1/chat"):
        return normalized + "/completions"
    return normalized + "/chat/completions"


def encode_image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")
    mime, _ = mimetypes.guess_type(path.name)
    if not mime:
        mime = "image/jpeg"
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_messages(system_prompt: str, user_prompt: str, image_path: str | None) -> list[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if image_path:
        user_content = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": encode_image_to_data_url(image_path),
                    "detail": "low",
                },
            },
        ]
    else:
        user_content = user_prompt

    messages.append({"role": "user", "content": user_content})
    return messages


def parse_reply(data: dict) -> str | None:
    choices = data.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        joined = "\n".join(x for x in texts if x)
        return joined or None
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal OpenAI-compatible caller."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_COMPATIBLE_MODEL", DEFAULT_MODEL_FOR_CALLING),
        help="Model name sent to the endpoint.",
    )
    parser.add_argument(
        "--prompt",
        default="请简单介绍一下你自己。",
        help="User prompt.",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="Optional system prompt.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional local image path for multimodal models.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="max_tokens in request payload.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="temperature in request payload.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override OPENAI_COMPATIBLE_API_KEY / OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Override OPENAI_COMPATIBLE_API_BASE / OPENAI_API_BASE.",
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Print full JSON response.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = resolve_api_key(args.api_key)
    api_base = resolve_api_base(args.api_base)
    if not api_key:
        print(
            "缺少 API Key，请设置 `OPENAI_COMPATIBLE_API_KEY` 或 `OPENAI_API_KEY`。",
            file=sys.stderr,
        )
        return 1

    payload = {
        "model": args.model,
        "messages": build_messages(args.system, args.prompt, args.image),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "n": 1,
    }
    request = urllib.request.Request(
        api_base,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            status_code = response.getcode()
            response_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        response_text = exc.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        print(f"请求失败: {exc}", file=sys.stderr)
        return 2

    try:
        data = json.loads(response_text)
    except ValueError:
        print("返回不是合法 JSON：", file=sys.stderr)
        print(response_text, file=sys.stderr)
        return 3

    print(f"[endpoint] {api_base}")
    print(f"[model] {args.model}")
    print(f"[http_status] {status_code}")

    if args.show_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))

    reply = parse_reply(data)
    if reply is not None:
        print("\n[reply]")
        print(reply)
        return 0 if 200 <= status_code < 300 else 4

    print("\n[raw_json]")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0 if 200 <= status_code < 300 else 4


if __name__ == "__main__":
    raise SystemExit(main())
