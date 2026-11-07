"""Generation-time text filler support for the ⟨V,T,F,T⟩ replay mode.

The filler block F reuses the construction of the scratchpad probe's
"Spaces" carrier (``vlmeval/probes/readout_carriers.py``, carrier name
``space_text``): a block built from the single-space literal whose token
count is matched to the core visual token count |V| of the sample's image.
The probe spliced ``|V|`` copies of the single-space token id at the token
level (``_insert_matched_text_carrier`` with ``literal=" "``); generation
prompts are plain text, so here the run of space characters is sized per
sample such that the model's tokenizer attributes exactly |V| tokens to it
in the surrounding prompt context. BPE vocabularies merge runs of spaces
(measured on the shared vLLM stack: up to 31 spaces per token for
Gemma-3-12B-it, up to 128 spaces per token for MiniCPM-o 4.5), so the
character count generally exceeds |V|; the in-context token count is the
quantity that must match.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable


SPACE_FILLER_LITERAL = " "
SPACE_FILLER_MODE = "image_text_filler_text"

# Search bounds for the per-sample character count. The lower bound assumes
# no merging (1 char = 1 token); observed merging is <= 128 chars/token on
# the supported models, so 400x leaves generous headroom.
_MAX_CHARS_FACTOR = 400


def tokenize_length(tokenizer: Any, text: str) -> int:
    """Token count of ``text`` without special-token additions.

    Both sides of every comparison go through this helper, so any constant
    special-token offset cancels; only the in-context difference matters.
    """
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def count_filler_tokens(
    tokenizer: Any,
    text_with_filler: str,
    text_without_filler: str,
) -> int:
    """In-context token count attributable to the filler block."""
    return tokenize_length(tokenizer, text_with_filler) - tokenize_length(
        tokenizer, text_without_filler
    )


def build_space_filler(
    count_fn: Callable[[str], int],
    target_tokens: int,
    *,
    tag: str = "",
) -> dict[str, Any]:
    """Find the smallest run of spaces whose in-context count is exactly
    ``target_tokens``.

    ``count_fn(filler_str)`` must return the in-context token count
    attributable to ``filler_str`` (see :func:`count_filler_tokens`). The
    count is non-decreasing in the number of characters for a pure-space
    filler; the search first expands exponentially, then binary-searches the
    minimal character count whose count is >= target, and finally requires
    exact equality. A miss is a hard error: the sample cannot be silently
    run with an unmatched filler.
    """
    target_tokens = int(target_tokens)
    if target_tokens <= 0:
        raise RuntimeError(f"[text-filler] non-positive target token count: {target_tokens} ({tag})")

    def probe(n_chars: int) -> int:
        return int(count_fn(SPACE_FILLER_LITERAL * n_chars))

    lo = 1
    hi = max(2, target_tokens)
    if probe(hi) < target_tokens:
        ceiling = target_tokens * _MAX_CHARS_FACTOR
        while hi < ceiling and probe(hi) < target_tokens:
            lo = hi + 1
            hi = min(hi * 2, ceiling)
        if probe(hi) < target_tokens:
            raise RuntimeError(
                f"[text-filler] no space run within {ceiling} chars reaches "
                f"{target_tokens} tokens ({tag})"
            )
    while lo < hi:
        mid = (lo + hi) // 2
        if probe(mid) < target_tokens:
            lo = mid + 1
        else:
            hi = mid
    n_chars = lo
    count = probe(n_chars)
    if count != target_tokens:
        near = {n: probe(n) for n in range(max(1, n_chars - 2), n_chars + 3)}
        raise RuntimeError(
            f"[text-filler] exact match unreachable ({tag}): target "
            f"{target_tokens} tokens, best candidate {n_chars} chars -> "
            f"{count} tokens; nearby counts {near}"
        )
    return {
        "literal": SPACE_FILLER_LITERAL,
        "chars": n_chars,
        "tokens": count,
        "target_tokens": target_tokens,
        "tag": tag,
    }


class TextFillerAudit:
    """Per-run audit log for the ⟨V,T,F,T⟩ mode.

    Enabled whenever the filler mode is active. Writes one JSON record per
    sample when ``REPLAY_TEXT_FILLER_AUDIT_FILE`` is set and prints a
    compact summary line periodically. Any engine-side mismatch raises, so
    a wrong run fails loudly instead of producing invalid predictions.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.path = os.environ.get("REPLAY_TEXT_FILLER_AUDIT_FILE", "").strip()
        self.verbose = os.environ.get("REPLAY_PROMPT_AUDIT_PRINT", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.records: list[dict[str, Any]] = []
        self._mismatch: Exception | None = None
        if self.path:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)

    def log_construction(self, record: dict[str, Any]) -> None:
        entry = {"phase": "construction", "ts": time.time(), "model": self.model_name, **record}
        self.records.append(entry)
        self._emit(entry)

    def log_engine(self, record: dict[str, Any]) -> None:
        entry = {"phase": "engine", "ts": time.time(), "model": self.model_name, **record}
        self.records.append(entry)
        self._emit(entry)
        diff = int(entry.get("filler_tokens_engine", entry.get("filler_tokens_local", 0))) - int(
            entry.get("visual_core_tokens", 0)
        )
        if entry.get("phase") == "engine" and diff != 0:
            message = (
                f"[text-filler] ENGINE AUDIT MISMATCH ({self.model_name}, "
                f"dataset={entry.get('dataset')}): filler engine tokens "
                f"{entry.get('filler_tokens_engine')} != visual core "
                f"{entry.get('visual_core_tokens')} (diff {diff})"
            )
            self._mismatch = RuntimeError(message)
            raise self._mismatch

    def _emit(self, entry: dict[str, Any]) -> None:
        if self.path:
            try:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError as err:
                print(f"[text-filler] audit write failed: {err}", flush=True)
        n = len(self.records)
        if self.verbose or n == 1 or n % 50 == 0:
            diffs = [
                int(r.get("filler_tokens_engine", r.get("filler_tokens_local", 0)))
                - int(r.get("visual_core_tokens", 0))
                for r in self.records
            ]
            print(
                f"[text-filler] audit n={n} dataset={entry.get('dataset')} "
                f"chars={entry.get('filler_chars')} "
                f"local={entry.get('filler_tokens_local')} "
                f"engine={entry.get('filler_tokens_engine')} "
                f"core={entry.get('visual_core_tokens')} "
                f"diff_min={min(diffs)} diff_max={max(diffs)}",
                flush=True,
            )
