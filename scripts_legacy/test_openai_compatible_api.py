#!/usr/bin/env python3
import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def resolve_api_key(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    return (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY_JUDGE", "").strip()
    )


def resolve_api_base(cli_value: str | None) -> str:
    base = cli_value
    if not base:
        base = (
            os.environ.get("OPENAI_API_BASE_JUDGE", "").strip()
            or os.environ.get("OPENAI_API_BASE", "").strip()
            or "https://api.openai.com/v1"
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


def build_payload(args: argparse.Namespace) -> dict:
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    return {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "n": 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use OpenAI-compatible chat/completions format to test a custom API."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="Model name sent in the OpenAI-compatible payload.",
    )
    parser.add_argument(
        "--prompt",
        default="Please reply with exactly: openai-compatible api is working",
        help="User prompt.",
    )
    parser.add_argument(
        "--system",
        default="You are a concise assistant.",
        help="Optional system prompt.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="max_tokens in the payload.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="temperature in the payload.",
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
        help="Override OPENAI_API_KEY / OPENAI_API_KEY_JUDGE.",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Override OPENAI_API_BASE_JUDGE / OPENAI_API_BASE.",
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Print the full JSON response.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = resolve_api_key(args.api_key)
    api_base = resolve_api_base(args.api_base)

    if not api_key:
        print(
            "[FATAL] OPENAI_API_KEY is empty. Please export OPENAI_API_KEY "
            "or pass --api-key.",
            file=sys.stderr,
        )
        return 1

    payload = build_payload(args)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    print(f"[INFO] endpoint: {api_base}")
    print(f"[INFO] model: {args.model}")
    print(f"[INFO] prompt: {args.prompt}")

    request = urllib.request.Request(
        api_base,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
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
        print(f"[ERROR] request failed: {exc}", file=sys.stderr)
        return 2

    print(f"[INFO] http_status: {status_code}")

    try:
        data = json.loads(response_text)
    except ValueError:
        print("[ERROR] response is not valid JSON:", file=sys.stderr)
        print(response_text, file=sys.stderr)
        return 3

    if args.show_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))

    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content")
    )
    if content is not None:
        print("\n[REPLY]")
        print(content)
        return 0 if 200 <= status_code < 300 else 4

    print("\n[WARN] no choices[0].message.content in response")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0 if 200 <= status_code < 300 else 4


if __name__ == "__main__":
    raise SystemExit(main())
