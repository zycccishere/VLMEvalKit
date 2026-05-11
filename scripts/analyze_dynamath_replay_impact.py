from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


FENCE = "`" * 3


def preprocess(text: str) -> str:
    text = str(text).strip().replace("\\n", "\n")
    if text.startswith(FENCE):
        lines = [x for x in text.splitlines() if not x.strip().startswith(FENCE)]
        text = "\n".join(lines).strip()
    return text


def extract_boxed(text: str) -> str | None:
    matches = re.findall(r"\\+boxed\s*\{([^{}]+)\}", text)
    if matches:
        return matches[-1].strip()
    return None


def iter_brace_chunks(text: str) -> list[str]:
    chunks: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue

        if ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                chunks.append(text[start: idx + 1])
                start = None

    return chunks


def extract_short_answer(text: str) -> tuple[str, str]:
    text = preprocess(text)
    candidates = [text]
    candidates.extend(reversed(iter_brace_chunks(text)))
    for cand in candidates:
        try:
            obj = json.loads(cand, strict=False)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        for key in ("short answer", "short_answer", "answer"):
            if key in obj and obj[key] is not None:
                return str(obj[key]).strip(), "json"

    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed, "boxed"
    return text, "fallback"


def transfer(token: str) -> float:
    if "π" in token:
        return float(token.split("π")[0]) * 3.141592653589793
    return float(token)


def parse_answer(answer: str, answer_type: str) -> tuple[bool, object]:
    answer = str(answer).strip()
    boxed = extract_boxed(answer)
    if boxed is not None:
        answer = boxed

    if answer_type == "float":
        answer = answer.replace(",", "")
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\s*π)?", answer)
        if match is None:
            return False, None
        token = match.group(0).replace(" ", "")
        try:
            return True, transfer(token)
        except Exception:
            return False, None

    if answer_type == "multiple choice":
        letters = re.findall(r"[A-E]", answer.upper())
        if len(set(letters)) == 1 and len(letters) >= 1:
            return True, letters[0]
        return False, None

    return True, answer


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def text_match(pred: str, gold: str) -> bool:
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    if not pred_norm or not gold_norm:
        return False
    return pred_norm == gold_norm or pred_norm in gold_norm or gold_norm in pred_norm


def robust_correct(prediction: str, answer: str, answer_type: str) -> tuple[bool, object, str]:
    short, source = extract_short_answer(prediction)
    ok, parsed = parse_answer(short, answer_type)
    if not ok:
        return False, short, source
    if answer_type == "float":
        return abs(float(parsed) - float(answer)) <= 0.001, parsed, source
    if answer_type == "multiple choice":
        return str(parsed) == str(answer), parsed, source
    return text_match(parsed, answer), parsed, source


def main() -> None:
    runs = Path("runs/by_setting")
    records = []

    for infer_xlsx in sorted(runs.rglob("*_DynaMath.xlsx")):
        if infer_xlsx.name.endswith("_gpt-4o-mini.xlsx"):
            continue

        parts = infer_xlsx.parts
        idx = parts.index("by_setting")
        policy = parts[idx + 1]
        mode = parts[idx + 2]
        model_key = parts[idx + 3]
        registry = parts[idx + 4]

        eval_xlsx = infer_xlsx.with_name(infer_xlsx.stem + "_gpt-4o-mini.xlsx")
        if not eval_xlsx.exists():
            continue

        infer_df = pd.read_excel(infer_xlsx)
        eval_df = pd.read_excel(eval_xlsx)

        robust_flags = []
        better = 0
        worse = 0
        for (_, infer_row), (_, eval_row) in zip(infer_df.iterrows(), eval_df.iterrows()):
            corr, _, _ = robust_correct(
                infer_row["prediction"],
                infer_row["answer"],
                infer_row["answer_type"],
            )
            robust_flags.append(corr)
            orig = bool(eval_row["correct"])
            if corr and not orig:
                better += 1
            elif orig and not corr:
                worse += 1

        robust_acc = sum(robust_flags) / len(robust_flags)
        orig_acc = float(eval_df["correct"].mean())
        records.append(
            {
                "policy": policy,
                "mode": mode,
                "model_key": model_key,
                "registry": registry,
                "rows": len(infer_df),
                "orig_acc": orig_acc,
                "robust_acc": robust_acc,
                "delta": robust_acc - orig_acc,
                "better": better,
                "worse": worse,
            }
        )

    records.sort(key=lambda x: (x["model_key"], x["policy"], x["mode"]))
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
