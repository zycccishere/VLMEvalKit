from __future__ import annotations

from .gpt import GPT4V
from .gemini import GeminiWrapper
from .claude import Claude3V
from .base import BaseAPI
from ..vlm.replay_policy import (
    apply_replay,
    canonicalize_replay_mode,
    is_noop_replay_mode,
    maybe_debug_print_replay,
    read_replay_config_from_env,
)
from ..vlm.qwen2_vl.replay_prompt_template import (
    apply_prompt_template_to_content,
    read_prompt_template_config_from_env,
)


class _ReplayAPIMixin:
    """Apply replay + prompt-template pipeline before API generation."""

    def _init_replay_mixin(self) -> None:
        self.replay_cfg = read_replay_config_from_env()
        self.prompt_template_cfg = read_prompt_template_config_from_env()
        self.template_on_last_replay_text = (
            str(getattr(self, "template_on_last_replay_text", "0")).strip().lower()
            in {"1", "true", "yes", "on"}
        )

    def _apply_prompt_template_to_content(self, content):
        return apply_prompt_template_to_content(content, self.prompt_template_cfg)

    def _apply_replay_to_content(self, content):
        replayed = apply_replay(
            content,
            mode=self.replay_cfg["mode"],
            repeat_times=self.replay_cfg["repeat_times"],
            image_copy_mode=self.replay_cfg["image_copy_mode"],
        )
        maybe_debug_print_replay(
            enabled=self.replay_cfg["debug"],
            mode=self.replay_cfg["mode"],
            before=content,
            after=replayed,
            tag=self.__class__.__name__,
        )
        return replayed

    def _apply_template_replay_pipeline(self, content):
        replay_mode = canonicalize_replay_mode(self.replay_cfg.get("mode", "image_text"))
        if self.template_on_last_replay_text and not is_noop_replay_mode(replay_mode):
            replayed = self._apply_replay_to_content(content)
            return self._apply_prompt_template_to_content(replayed)
        templated = self._apply_prompt_template_to_content(content)
        return self._apply_replay_to_content(templated)


class GPT4VReplay(_ReplayAPIMixin, GPT4V):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_replay_mixin()

    def generate_inner(self, inputs, **kwargs):
        patched = self._apply_template_replay_pipeline(inputs)
        return super().generate_inner(patched, **kwargs)


class GeminiReplay(_ReplayAPIMixin, GeminiWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_replay_mixin()

    def generate_inner(self, inputs, **kwargs):
        patched = self._apply_template_replay_pipeline(inputs)
        return super().generate_inner(patched, **kwargs)


class ClaudeReplay(_ReplayAPIMixin, Claude3V):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_replay_mixin()

    def generate_inner(self, inputs, **kwargs):
        patched = self._apply_template_replay_pipeline(inputs)
        return super().generate_inner(patched, **kwargs)
