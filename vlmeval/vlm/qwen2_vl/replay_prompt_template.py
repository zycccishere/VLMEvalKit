import os


PROMPT_TEMPLATE_IDENTITY = "identity"
PROMPT_TEMPLATE_DIRECTLY_ANSWER = "directly_answer"

DEFAULT_PROMPT_TEMPLATES = {
    PROMPT_TEMPLATE_IDENTITY: "{problem}",
    PROMPT_TEMPLATE_DIRECTLY_ANSWER: (
        "{problem}\n"
        "Answer directly with a single word or short phrase.\n"
        "Do not output any explanation, derivation, words, or extra symbols."
    ),
}

_TEXT_MARKERS_TO_SKIP = {"<video frames start>", "<video frames end>"}
_DIRECT_TEMPLATE_STRIP_SUFFIXES = {
    "VisuLogic": [
        "\nSolve the complex visual logical reasoning problem through step-by-step reasoning."
        "Think about the reasoning process first "
        "and answer the question following this format: Answer: \\boxed{$LETTER}",
    ],
    "VisualPuzzles": [
        "\nSolve the multiple-choice question and then answer with the option letter from the given choices. "
        "The last line of your response should be of the following format:"
        "'Answer: $LETTER' (without quotes) where LETTER is one of options. "
        "Think step by step before answering.",
    ],
}


def _text_keys(item: dict[str, str], preferred_key: str = "text") -> list[str]:
    keys = []
    if preferred_key in item:
        keys.append(preferred_key)
    for key in ("text", "value"):
        if key in item and key not in keys:
            keys.append(key)
    return keys


def _text_value(item: dict[str, str], preferred_key: str = "text") -> str:
    for key in _text_keys(item, preferred_key=preferred_key):
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _set_text_value(item: dict[str, str], value: str, preferred_key: str = "text") -> None:
    keys = _text_keys(item, preferred_key=preferred_key)
    if not keys:
        keys = [preferred_key]
    for key in keys:
        item[key] = value


def read_prompt_template_config_from_env() -> dict[str, str]:
    template_name = (
        os.environ.get("REPLAY_PROMPT_TEMPLATE_NAME")
        or os.environ.get("PROMPT_TEMPLATE_NAME")
        or PROMPT_TEMPLATE_IDENTITY
    ).strip()

    template_file = (os.environ.get("REPLAY_PROMPT_TEMPLATE_FILE") or "").strip()
    inline_template = os.environ.get("REPLAY_PROMPT_TEMPLATE")

    if template_file:
        try:
            with open(template_file, "r", encoding="utf-8") as f:
                template_text = f.read()
            source = "file"
        except Exception as err:
            print(
                f"[prompt-template] failed to read file '{template_file}': {err}. "
                f"Falling back to template name '{template_name}'.",
                flush=True,
            )
            template_text = DEFAULT_PROMPT_TEMPLATES.get(
                template_name,
                DEFAULT_PROMPT_TEMPLATES[PROMPT_TEMPLATE_IDENTITY],
            )
            source = "named_fallback"
    elif inline_template and inline_template.strip():
        template_text = inline_template
        source = "inline"
    else:
        template_text = DEFAULT_PROMPT_TEMPLATES.get(
            template_name,
            DEFAULT_PROMPT_TEMPLATES[PROMPT_TEMPLATE_IDENTITY],
        )
        source = "named"

    return {
        "name": template_name,
        "template": template_text,
        "source": source,
    }


def _match_dataset_name(dataset: str | None) -> str | None:
    if not dataset:
        return None
    dataset_lower = str(dataset).lower()
    for name in ("VisuLogic", "LogicVista", "VisualPuzzles"):
        if name.lower() in dataset_lower:
            return name
    return None


def strip_dataset_prompt_template_for_direct_answer(problem: str, dataset: str | None = None) -> str:
    base = str(problem).strip()
    matched_name = _match_dataset_name(dataset)
    if matched_name is None:
        return base

    for suffix in _DIRECT_TEMPLATE_STRIP_SUFFIXES.get(matched_name, []):
        if base.endswith(suffix):
            return base[: -len(suffix)].rstrip()
    return base


def strip_prompt_template_from_content_for_direct_answer(
    content: list[dict[str, str]],
    dataset: str | None = None,
    text_key: str = "text",
) -> list[dict[str, str]]:
    output = []
    for item in content:
        if not isinstance(item, dict):
            output.append(item)
            continue
        if item.get("type") != "text":
            output.append(item)
            continue
        new_item = dict(item)
        _set_text_value(
            new_item,
            strip_dataset_prompt_template_for_direct_answer(
                _text_value(item, preferred_key=text_key),
                dataset=dataset,
            ),
            preferred_key=text_key,
        )
        output.append(new_item)
    return output


def render_prompt_with_template(
    problem: str,
    template_cfg: dict[str, str],
    dataset: str | None = None,
) -> str:
    if template_cfg.get("name") == PROMPT_TEMPLATE_DIRECTLY_ANSWER:
        line1 = "Answer directly with a single word or short phrase."
        line2 = "Do not output any explanation, derivation, words, or extra symbols."
        base = strip_dataset_prompt_template_for_direct_answer(problem, dataset=dataset)
        if line1 not in base:
            base = f"{base}\n{line1}" if base else line1
        if line2 not in base:
            base = f"{base}\n{line2}" if base else line2
        return base

    template_text = template_cfg.get("template", "{problem}")
    # Use direct token replacement instead of str.format to avoid breaking
    # templates that intentionally contain braces, e.g. \boxed{<ANSWER>}.
    return template_text.replace("{problem}", problem).replace("{question}", problem)


def apply_prompt_template_to_content(
    content: list[dict[str, str]],
    template_cfg: dict[str, str],
    dataset: str | None = None,
    text_key: str = "text",
) -> list[dict[str, str]]:
    if not content:
        return content

    if template_cfg.get("template", "{problem}") == "{problem}":
        return content

    target_idx = None
    for idx, item in enumerate(content):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "text":
            continue
        if _text_value(item, preferred_key=text_key) in _TEXT_MARKERS_TO_SKIP:
            continue
        target_idx = idx

    if target_idx is None:
        return content

    output = []
    for idx, item in enumerate(content):
        if idx != target_idx:
            output.append(item)
            continue
        new_item = dict(item)
        _set_text_value(
            new_item,
            render_prompt_with_template(
                _text_value(item, preferred_key=text_key),
                template_cfg,
                dataset=dataset,
            ),
            preferred_key=text_key,
        )
        output.append(new_item)
    return output
