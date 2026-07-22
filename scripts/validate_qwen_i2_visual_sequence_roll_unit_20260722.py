#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from vlmeval.vlm.qwen_i2_visual_sequence_roll import (
    VISUAL_SEQUENCE_ROLL_RIGHT_1,
    QwenI2VisualSequenceRoll,
)


class FakeVisual(torch.nn.Module):
    spatial_merge_size = 2

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        return hidden_states


class FakeLanguageModel(torch.nn.Module):
    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return inputs_embeds


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(visual=FakeVisual(), language_model=FakeLanguageModel())


def main() -> int:
    model = FakeModel()
    image = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    sentinel = torch.cat([image, image.clone()], dim=0)
    grid_thw = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    expected = sentinel.clone()
    expected[4:] = sentinel[[7, 4, 5, 6]]

    with tempfile.TemporaryDirectory() as tmp:
        dump_dir = Path(tmp) / "visual_sequence_npz"
        controller = QwenI2VisualSequenceRoll(model, dump_dir=dump_dir, raw_dump_limit=1)
        with controller.sample(
            case_id="sentinel",
            intervention="baseline",
            enabled=True,
            roll_enabled=False,
            image1_positions=[0, 1, 2, 3],
            image2_positions=[4, 5, 6, 7],
        ) as baseline_record:
            baseline = model.model.visual(sentinel.clone(), grid_thw=grid_thw)
            model.model.language_model(
                inputs_embeds=baseline.unsqueeze(0),
                position_ids=torch.arange(8).reshape(1, 8),
                attention_mask=torch.ones((1, 8), dtype=torch.long),
            )
        with controller.sample(
            case_id="sentinel",
            intervention=VISUAL_SEQUENCE_ROLL_RIGHT_1,
            enabled=True,
            roll_enabled=True,
            image1_positions=[0, 1, 2, 3],
            image2_positions=[4, 5, 6, 7],
        ) as roll_record:
            rolled = model.model.visual(sentinel.clone(), grid_thw=grid_thw)
            model.model.language_model(
                inputs_embeds=rolled.unsqueeze(0),
                position_ids=torch.arange(8).reshape(1, 8),
                attention_mask=torch.ones((1, 8), dtype=torch.long),
            )
        controller.close()

        raw_path = dump_dir.parent / str(roll_record["raw_npz_path"])
        raw = np.load(raw_path)
        source = np.asarray(raw["source_index_for_output"], dtype=np.int64)
        checks = {
            "baseline_noop": torch.equal(baseline, sentinel) and baseline_record["apply_count"] == 0,
            "full_output_exact": torch.equal(rolled, expected),
            "i1_unchanged": torch.equal(rolled[:4], sentinel[:4]),
            "i2_1234_to_4123": torch.equal(rolled[4:], sentinel[[7, 4, 5, 6]]),
            "source_mapping": source.tolist() == [3, 0, 1, 2],
            "raw_i1_exact": np.array_equal(raw["image1_before"], raw["image1_after"]),
            "raw_i2_recomputed": np.array_equal(raw["image2_after"], raw["image2_before"][source]),
            "raw_i2_injected": np.array_equal(raw["llm_injected_image2"], raw["image2_after"]),
            "stage_exact": roll_record["stage"] == "qwen_post_spatial_merger_pre_llm_injection",
            "single_apply": roll_record["apply_count"] == 1,
            "single_injection": roll_record["language_injection_hook_count"] == 1,
            "pixel_equivalent_na": roll_record["pixel_equivalent"] is None,
        }
    result = {"ok": all(checks.values()), "checks": checks}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
