import copy
import json
import os
from typing import Any


REPLAY_IMAGE_TEXT = "image_text"
REPLAY_NONE = "none"  # backward-compatible alias for image_text
REPLAY_TEXT_IMAGE = "text_image"
# Canonical names (ordered by modal sequence).
REPLAY_IMAGE_TEXT_TEXT = "image_text_text"
REPLAY_IMAGE_TEXT_IMAGE_TEXT = "image_text_image_text"
REPLAY_IMAGE_TEXT_IMAGE = "image_text_image"
REPLAY_IMAGE_IMAGE_TEXT = "image_image_text"
# Multi-image-copy SLP tiers: <V,T,V,...,V,T> with 2 / 3 image copies between
# the two text copies. Combined with the `blank` image transform on image
# positions 2..N+1 these yield <V,T,box,box,T> / <V,T,box,box,box,T>.
REPLAY_IMAGE_TEXT_IMAGE_IMAGE_TEXT = "image_text_image_image_text"
REPLAY_IMAGE_TEXT_IMAGE_IMAGE_IMAGE_TEXT = "image_text_image_image_image_text"
# ⟨V,T,F,T⟩ generation-time text filler. The filler block F is model-aware
# (its in-context token count is matched per sample to the image's core
# visual token count), so this mode is canonicalized here but constructed
# inside the model wrappers (see vlmeval/vlm/replay_text_filler.py);
# apply_replay refuses it loudly instead of degrading to a no-op.
REPLAY_IMAGE_TEXT_FILLER_TEXT = "image_text_filler_text"

# Backward-compatible aliases.
REPLAY_PROMPT = REPLAY_IMAGE_TEXT_TEXT
REPLAY_FULL_SEQUENCE = REPLAY_IMAGE_TEXT_IMAGE_TEXT
REPLAY_IMAGE = REPLAY_IMAGE_TEXT_IMAGE

CANONICAL_REPLAY_MODES = {
    REPLAY_IMAGE_TEXT,
    REPLAY_TEXT_IMAGE,
    REPLAY_IMAGE_TEXT_TEXT,
    REPLAY_IMAGE_TEXT_IMAGE_TEXT,
    REPLAY_IMAGE_TEXT_IMAGE,
    REPLAY_IMAGE_IMAGE_TEXT,
    REPLAY_IMAGE_TEXT_IMAGE_IMAGE_TEXT,
    REPLAY_IMAGE_TEXT_IMAGE_IMAGE_IMAGE_TEXT,
    REPLAY_IMAGE_TEXT_FILLER_TEXT,
}
SUPPORTED_REPLAY_MODES = CANONICAL_REPLAY_MODES | {REPLAY_NONE}

REPLAY_MODE_ALIASES = {
    REPLAY_NONE: REPLAY_IMAGE_TEXT,
    "prompt_replay": REPLAY_IMAGE_TEXT_TEXT,
    "full_sequence_replay": REPLAY_IMAGE_TEXT_IMAGE_TEXT,
    "image_replay": REPLAY_IMAGE_TEXT_IMAGE,
    "image_image_text_replay": REPLAY_IMAGE_IMAGE_TEXT,
}

# Number of image copies inserted between the original pair and the trailing
# text copy for the image-copy SLP family. image_text_image_text is the
# single-copy member (kept on its legacy full-sequence implementation).
REPLAY_MODE_IMAGE_COPY_COUNTS = {
    REPLAY_IMAGE_TEXT_IMAGE_TEXT: 1,
    REPLAY_IMAGE_TEXT_IMAGE_IMAGE_TEXT: 2,
    REPLAY_IMAGE_TEXT_IMAGE_IMAGE_IMAGE_TEXT: 3,
}


def canonicalize_replay_mode(mode: str | None) -> str:
    raw = str(mode or REPLAY_IMAGE_TEXT).strip().lower()
    raw = REPLAY_MODE_ALIASES.get(raw, raw)
    if raw not in CANONICAL_REPLAY_MODES:
        return REPLAY_IMAGE_TEXT
    return raw


def is_noop_replay_mode(mode: str | None) -> bool:
    return canonicalize_replay_mode(mode) == REPLAY_IMAGE_TEXT


def replay_mode_image_copy_count(mode: str | None) -> int:
    """Image copies inserted between the original pair and the trailing text copy.

    Returns 1 for ``image_text_image_text`` (legacy single-copy SLP), 2/3 for
    the multi-image-copy tiers, and 0 for every other mode. Callers use this
    to derive the blank-transform target positions ``2..N+1``.
    """
    return REPLAY_MODE_IMAGE_COPY_COUNTS.get(canonicalize_replay_mode(mode), 0)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        return int(raw)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, str(default)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def read_replay_config_from_env() -> dict[str, Any]:
    mode = canonicalize_replay_mode(os.environ.get("REPLAY_MODE", REPLAY_IMAGE_TEXT))

    repeat_times = max(1, _env_int("REPLAY_TIMES", 1))
    debug = _env_bool("REPLAY_DEBUG", False)
    image_copy_mode = os.environ.get("REPLAY_IMAGE_COPY_MODE", "reuse_path").strip().lower()
    if image_copy_mode not in {"reuse_path", "deepcopy"}:
        image_copy_mode = "reuse_path"

    return {
        "mode": mode,
        "repeat_times": repeat_times,
        "debug": debug,
        "image_copy_mode": image_copy_mode,
    }


def _copy_item(item: dict[str, Any], image_copy_mode: str) -> dict[str, Any]:
    if image_copy_mode == "deepcopy":
        return copy.deepcopy(item)
    return dict(item)


def _split_modal_items(content: list[dict[str, Any]], image_copy_mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    text_items = []
    vision_items = []
    other_items = []
    for item in content:
        copied = _copy_item(item, image_copy_mode)
        item_type = copied.get("type")
        if item_type == "text":
            text_items.append(copied)
        elif item_type in {"image", "video"}:
            vision_items.append(copied)
        else:
            other_items.append(copied)
    return text_items, vision_items, other_items


def _repeat_prompt_once(content: list[dict[str, Any]], image_copy_mode: str) -> list[dict[str, Any]]:
    text_items, _, _ = _split_modal_items(content, image_copy_mode)
    return content + text_items


def _repeat_full_sequence_once(content: list[dict[str, Any]], image_copy_mode: str) -> list[dict[str, Any]]:
    copied = [_copy_item(x, image_copy_mode) for x in content]
    return content + copied


def _repeat_image_once(content: list[dict[str, Any]], image_copy_mode: str) -> list[dict[str, Any]]:
    _, vision_items, _ = _split_modal_items(content, image_copy_mode)
    return content + vision_items


def _repeat_prompt_with_image_copies_once(
    content: list[dict[str, Any]],
    image_copy_mode: str,
    image_copies: int,
) -> list[dict[str, Any]]:
    text_items, vision_items, _ = _split_modal_items(content, image_copy_mode)
    extra_vision: list[dict[str, Any]] = []
    for _ in range(image_copies):
        extra_vision.extend(_copy_item(item, image_copy_mode) for item in vision_items)
    return content + extra_vision + text_items


def _reorder_to_text_image_once(content: list[dict[str, Any]], image_copy_mode: str) -> list[dict[str, Any]]:
    text_items, vision_items, other_items = _split_modal_items(content, image_copy_mode)
    return text_items + vision_items + other_items


def _reorder_to_image_image_text_once(content: list[dict[str, Any]], image_copy_mode: str) -> list[dict[str, Any]]:
    text_items, vision_items, other_items = _split_modal_items(content, image_copy_mode)
    return vision_items + [_copy_item(x, image_copy_mode) for x in vision_items] + text_items + other_items


def apply_replay(content: list[dict[str, Any]], mode: str, repeat_times: int = 1, image_copy_mode: str = "reuse_path") -> list[dict[str, Any]]:
    mode = canonicalize_replay_mode(mode)
    if mode == REPLAY_IMAGE_TEXT_FILLER_TEXT:
        raise RuntimeError(
            "image_text_filler_text requires model-aware filler construction "
            "(in-context token matching against the sample's visual core "
            "count); the wrappers intercept this mode before apply_replay"
        )
    if is_noop_replay_mode(mode) or repeat_times <= 0:
        return content

    out = list(content)
    for _ in range(repeat_times):
        if mode == REPLAY_PROMPT:
            out = _repeat_prompt_once(out, image_copy_mode=image_copy_mode)
        elif mode == REPLAY_FULL_SEQUENCE:
            out = _repeat_full_sequence_once(out, image_copy_mode=image_copy_mode)
        elif mode == REPLAY_IMAGE:
            out = _repeat_image_once(out, image_copy_mode=image_copy_mode)
        elif mode == REPLAY_TEXT_IMAGE:
            out = _reorder_to_text_image_once(out, image_copy_mode=image_copy_mode)
        elif mode == REPLAY_IMAGE_IMAGE_TEXT:
            out = _reorder_to_image_image_text_once(out, image_copy_mode=image_copy_mode)
        elif mode in REPLAY_MODE_IMAGE_COPY_COUNTS:
            out = _repeat_prompt_with_image_copies_once(
                out,
                image_copy_mode=image_copy_mode,
                image_copies=REPLAY_MODE_IMAGE_COPY_COUNTS[mode],
            )
        else:
            return content
    return out


def count_modalities(content: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"text": 0, "image": 0, "video": 0}
    for item in content:
        t = item.get("type")
        if t in counts:
            counts[t] += 1
    return counts


def maybe_debug_print_replay(enabled: bool, mode: str, before: list[dict[str, Any]], after: list[dict[str, Any]], tag: str = "replay") -> None:
    if not enabled:
        return

    before_counts = count_modalities(before)
    after_counts = count_modalities(after)
    payload = {
        "tag": tag,
        "mode": canonicalize_replay_mode(mode),
        "before_counts": before_counts,
        "after_counts": after_counts,
        "before_content": before,
        "after_content": after,
    }
    try:
        print(f"[REPLAY_DEBUG] {json.dumps(payload, ensure_ascii=False)}", flush=True)
    except Exception:
        print(
            f"[REPLAY_DEBUG] mode={canonicalize_replay_mode(mode)} before={before_counts} after={after_counts}",
            flush=True,
        )
