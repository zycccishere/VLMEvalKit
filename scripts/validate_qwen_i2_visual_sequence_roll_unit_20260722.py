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
    processed_image_pair_contract,
    qwen_attention_return_arity,
    tensor_sha256,
)
from scripts.validate_qwen25vl_shift_flow_smoke_20260629 import (
    decode_raw_float_bytes,
    raw_byte_sha256,
    validate_processor_pair_raw,
)


class FakeVisual(torch.nn.Module):
    spatial_merge_size = 2

    def __init__(self, *, return_container: bool = False) -> None:
        super().__init__()
        self.return_container = return_container

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
        if self.return_container:
            return SimpleNamespace(pooler_output=hidden_states)
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
    def __init__(self, *, return_container: bool = False) -> None:
        super().__init__()
        self.model = SimpleNamespace(
            visual=FakeVisual(return_container=return_container),
            language_model=FakeLanguageModel(),
        )


def raises(expected: type[BaseException], fn) -> bool:
    try:
        fn()
    except expected:
        return True
    return False


def main() -> int:
    model = FakeModel()
    image1 = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
    image2 = image1 + 100.0
    sentinel = torch.cat([image1, image2], dim=0)
    grid_thw = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    expected = sentinel.clone()
    expected[4:] = sentinel[[7, 4, 5, 6]]
    text_embeddings = torch.full((2, 3), -7.0)
    image1_positions = [0, 1, 2, 3]
    image2_positions = [6, 7, 8, 9]
    position_ids = torch.arange(10).reshape(1, 10)
    attention_mask = torch.ones((1, 10), dtype=torch.long)

    patch_rows = torch.arange(16 * 6, dtype=torch.float32).reshape(16, 6)
    exact_pair_inputs = {
        "pixel_values": torch.cat([patch_rows, patch_rows.clone()], dim=0),
        "image_grid_thw": grid_thw.clone(),
    }
    exact_pair_contract = processed_image_pair_contract(exact_pair_inputs, spatial_merge_size=2)
    unequal_pair_inputs = {
        "pixel_values": torch.cat([patch_rows, patch_rows + 1], dim=0),
        "image_grid_thw": grid_thw.clone(),
    }
    unequal_pair_contract = processed_image_pair_contract(unequal_pair_inputs, spatial_merge_size=2)

    with tempfile.TemporaryDirectory() as tmp:
        wrong_grid = np.asarray([[1, 4, 4], [4, 2, 2]], dtype=np.int64)
        wrong_grid_raw_path = Path(tmp) / "equal_product_different_grid.npz"
        np.savez_compressed(
            wrong_grid_raw_path,
            pixel_values=exact_pair_inputs["pixel_values"].numpy(),
            image_grid_thw=wrong_grid,
        )
        wrong_grid_contract = {
            "grid_thw": wrong_grid.tolist(),
            "spatial_merge_size": 2,
            "patch_row_counts": [16, 16],
            "pixel_values_shape": [32, 6],
            "image_patch_row_shapes": [[16, 6], [16, 6]],
            "same_grid": True,
            "patch_rows_exact": True,
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "image1_sha256": tensor_sha256(patch_rows),
            "image2_sha256": tensor_sha256(patch_rows),
        }
        wrong_grid_raw_ok, _ = validate_processor_pair_raw(
            wrong_grid_raw_path,
            wrong_grid_contract,
            image1_tokens=4,
            image2_tokens=4,
        )

        dump_dir = Path(tmp) / "visual_sequence_npz"
        controller = QwenI2VisualSequenceRoll(model, dump_dir=dump_dir, raw_dump_limit=1)
        with controller.sample(
            case_id="sentinel",
            intervention="baseline",
            enabled=True,
            roll_enabled=False,
            image1_positions=image1_positions,
            image2_positions=image2_positions,
        ) as baseline_record:
            baseline_visual = model.model.visual(sentinel.clone(), grid_thw=grid_thw)
            baseline = torch.cat([baseline_visual[:4], text_embeddings, baseline_visual[4:]], dim=0)
            model.model.language_model(
                inputs_embeds=baseline.unsqueeze(0),
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
        with controller.sample(
            case_id="sentinel",
            intervention=VISUAL_SEQUENCE_ROLL_RIGHT_1,
            enabled=True,
            roll_enabled=True,
            image1_positions=image1_positions,
            image2_positions=image2_positions,
        ) as roll_record:
            rolled_visual = model.model.visual(sentinel.clone(), grid_thw=grid_thw)
            rolled = torch.cat([rolled_visual[:4], text_embeddings, rolled_visual[4:]], dim=0)
            model.model.language_model(
                inputs_embeds=rolled.unsqueeze(0),
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
        controller.close()

        container_model = FakeModel(return_container=True)
        container_controller = QwenI2VisualSequenceRoll(
            container_model,
            dump_dir=Path(tmp) / "container_npz",
            raw_dump_limit=0,
        )
        with container_controller.sample(
            case_id="container-sentinel",
            intervention=VISUAL_SEQUENCE_ROLL_RIGHT_1,
            enabled=True,
            roll_enabled=True,
            image1_positions=image1_positions,
            image2_positions=image2_positions,
        ) as container_record:
            container_output = container_model.model.visual(sentinel.clone(), grid_thw=grid_thw)
            container_llm = torch.cat(
                [container_output.pooler_output[:4], text_embeddings, container_output.pooler_output[4:]],
                dim=0,
            )
            container_model.model.language_model(
                inputs_embeds=container_llm.unsqueeze(0),
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
        container_controller.close()

        raw_path = dump_dir.parent / str(roll_record["raw_npz_path"])
        raw = np.load(raw_path)
        source = np.asarray(raw["source_index_for_output"], dtype=np.int64)
        raw_binding_fields = {
            "image1_before_raw_bytes": ("image1_before_sha256", "image1_before"),
            "image1_after_raw_bytes": ("image1_after_sha256", "image1_after"),
            "image2_before_raw_bytes": ("image2_before_sha256", "image2_before"),
            "image2_after_raw_bytes": ("image2_after_sha256", "image2_after"),
            "llm_injected_image1_raw_bytes": ("llm_i1_sha256", "llm_injected_image1"),
            "llm_injected_image2_raw_bytes": ("llm_i2_sha256", "llm_injected_image2"),
        }
        raw_hashes_and_values_bound = all(
            raw_byte_sha256(raw[raw_field]) == roll_record[summary_field]
            and np.array_equal(
                decode_raw_float_bytes(raw[raw_field], "torch.float32", tuple(raw[float_field].shape)),
                raw[float_field],
            )
            for raw_field, (summary_field, float_field) in raw_binding_fields.items()
        )
        checks = {
            "baseline_noop": torch.equal(baseline_visual, sentinel) and baseline_record["apply_count"] == 0,
            "full_output_exact": torch.equal(rolled_visual, expected),
            "i1_unchanged": torch.equal(rolled[:4], sentinel[:4]),
            "i2_1234_to_4123": torch.equal(rolled[6:], sentinel[[7, 4, 5, 6]]),
            "text_span_unchanged": torch.equal(rolled[4:6], text_embeddings),
            "all_non_i2_embeddings_exact": roll_record["llm_non_i2_sha256"]
            == tensor_sha256(torch.cat([sentinel[:4], text_embeddings], dim=0)),
            "source_mapping": source.tolist() == [3, 0, 1, 2],
            "raw_i1_exact": np.array_equal(raw["image1_before"], raw["image1_after"]),
            "raw_i2_recomputed": np.array_equal(raw["image2_after"], raw["image2_before"][source]),
            "raw_i2_injected": np.array_equal(raw["llm_injected_image2"], raw["image2_after"]),
            "raw_hashes_and_values_bound": raw_hashes_and_values_bound,
            "nonidentical_post_merger_supported": roll_record["repeated_image_embeddings_exact"] is False,
            "post_merger_difference_recorded": roll_record["repeated_image_embeddings_max_abs_diff"] == 100.0,
            "post_merger_mean_difference_recorded": roll_record["repeated_image_embeddings_mean_abs_diff"] == 100.0,
            "post_merger_relative_metrics_finite": np.isfinite(
                roll_record["repeated_image_embeddings_relative_rms"]
            )
            and np.isfinite(roll_record["repeated_image_embeddings_mean_cosine"]),
            "container_output_path": torch.equal(container_output.pooler_output, expected)
            and container_record["visual_output_kind"] == "base_model_output_with_pooling"
            and container_record["llm_i2_injection_exact"] is True,
            "processor_exact_pair": exact_pair_contract["validated"] is True
            and exact_pair_contract["patch_rows_exact"] is True
            and exact_pair_contract["patch_row_counts"] == [16, 16],
            "processor_unequal_pair_rejected": unequal_pair_contract["validated"] is False
            and unequal_pair_contract["patch_rows_exact"] is False,
            "processor_3d_rejected": raises(
                TypeError,
                lambda: processed_image_pair_contract(
                    {
                        "pixel_values": exact_pair_inputs["pixel_values"].reshape(32, 1, 6),
                        "image_grid_thw": grid_thw,
                    },
                    spatial_merge_size=2,
                ),
            ),
            "processor_float_grid_rejected": raises(
                TypeError,
                lambda: processed_image_pair_contract(
                    {"pixel_values": exact_pair_inputs["pixel_values"], "image_grid_thw": grid_thw.float()},
                    spatial_merge_size=2,
                ),
            ),
            "processor_nonpositive_grid_rejected": raises(
                ValueError,
                lambda: processed_image_pair_contract(
                    {
                        "pixel_values": exact_pair_inputs["pixel_values"],
                        "image_grid_thw": torch.tensor([[1, 4, 4], [0, 4, 4]]),
                    },
                    spatial_merge_size=2,
                ),
            ),
            "processor_nondivisible_grid_rejected": raises(
                ValueError,
                lambda: processed_image_pair_contract(
                    {
                        "pixel_values": torch.zeros((18, 6)),
                        "image_grid_thw": torch.tensor([[1, 3, 3], [1, 3, 3]]),
                    },
                    spatial_merge_size=2,
                ),
            ),
            "validator_equal_product_different_grid_rejected": not wrong_grid_raw_ok,
            "attention_arity_transformers_4533": qwen_attention_return_arity("4.53.3") == 3,
            "attention_arity_transformers_550": qwen_attention_return_arity("5.5.0") == 2,
            "attention_arity_unvalidated_fails_closed": raises(
                RuntimeError,
                lambda: qwen_attention_return_arity("4.57.3"),
            ),
            "stage_exact": roll_record["stage"] == "qwen_post_spatial_merger_pre_llm_injection",
            "single_apply": roll_record["apply_count"] == 1,
            "single_injection": roll_record["language_injection_hook_count"] == 1,
            "pixel_equivalent_na": roll_record["pixel_equivalent"] is None,
        }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {"ok": all(checks.values()), "checks": checks}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
