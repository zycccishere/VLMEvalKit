import re


def tail_tokens_for_judge(text, max_tokens=512):
    text = "" if text is None else str(text)
    text = text.strip()
    if not text:
        return text

    parts = re.findall(r"\S+|\s+", text)
    token_idx = [i for i, part in enumerate(parts) if not part.isspace()]
    if len(token_idx) <= max_tokens:
        return text

    start = token_idx[-max_tokens]
    tail = "".join(parts[start:]).lstrip()
    return tail
