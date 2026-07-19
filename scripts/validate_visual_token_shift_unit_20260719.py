#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

MODULE_PATH = Path(__file__).resolve().parents[1] / "vlmeval" / "vlm" / "replay_visual_token_shift.py"
SPEC = importlib.util.spec_from_file_location("replay_visual_token_shift_direct", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
SHIFT_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHIFT_MODULE)
VisualTokenShiftController = SHIFT_MODULE.VisualTokenShiftController
roll_visual_token_blocks = SHIFT_MODULE.roll_visual_token_blocks
validate_iqiq_topology = SHIFT_MODULE.validate_iqiq_topology


class FakeQwenVisual(torch.nn.Module):
    spatial_merge_size = 2

    def forward(self, merged_output: torch.Tensor, *, grid_thw: torch.Tensor) -> BaseModelOutputWithPooling:
        return BaseModelOutputWithPooling(
            last_hidden_state=merged_output.clone() + 1000.0,
            pooler_output=merged_output,
        )


class FakeQwenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = FakeQwenVisual()


class FakeMiniCPMModel:
    def __init__(self, states: torch.Tensor) -> None:
        self.states = states
        self.config = SimpleNamespace(query_num=64)

    def get_vision_embedding(self, data):
        return [self.states.clone()]


class FakeMiniCPMProcessor:
    def __init__(self, data) -> None:
        self.data = data
        self.tokenizer = "tokenizer-before"

    def __call__(self, *args, **kwargs):
        return self.data


def configure(output_dir: Path, mode: str) -> None:
    os.environ["REPLAY_VISUAL_TOKEN_SHIFT"] = mode
    os.environ["REPLAY_VISUAL_TOKEN_SHIFT_TARGET_POSITION"] = "2"
    os.environ["REPLAY_VISUAL_TOKEN_SHIFT_STRICT"] = "1"
    os.environ["REPLAY_VISUAL_TOKEN_SHIFT_RAW_DUMP"] = "1"
    os.environ["REPLAY_VISUAL_TOKEN_SHIFT_DUMP_SAMPLES"] = "8"
    os.environ["REPLAY_VISUAL_TOKEN_SHIFT_DUMP_DIR"] = str(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sentinel = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    right, right_record = roll_visual_token_blocks(sentinel, shift=1)
    left, left_record = roll_visual_token_blocks(sentinel, shift=-1)
    checks = {
        "pure_right_1234_to_4123": right.flatten().tolist() == [4.0, 1.0, 2.0, 3.0],
        "pure_left_1234_to_2341": left.flatten().tolist() == [2.0, 3.0, 4.0, 1.0],
        "pure_right_exact": right_record["exact_roll_verified"] and right_record["max_abs_error"] == 0.0,
        "pure_left_exact": left_record["exact_roll_verified"] and left_record["max_abs_error"] == 0.0,
    }
    iq = [{"type": "image", "value": "/tmp/i.png"}, {"type": "text", "value": "Q"}]
    iqiq = iq + [dict(item) for item in iq]
    topology = validate_iqiq_topology(
        original=iq,
        replayed=iqiq,
        replay_mode="image_text_image_text",
        repeat_times=1,
        image_transform_name="baseline",
        target_image_position=2,
    )
    checks["iqiq_topology_validated"] = topology["validated"]
    try:
        validate_iqiq_topology(
            original=iqiq,
            replayed=iqiq + iqiq,
            replay_mode="image_text_image_text",
            repeat_times=1,
            image_transform_name="baseline",
            target_image_position=2,
        )
    except ValueError:
        checks["multi_image_source_fails_closed"] = True
    else:
        checks["multi_image_source_fails_closed"] = False

    qwen_dir = args.output_dir / "qwen"
    configure(qwen_dir, "roll_right_1")
    qwen_model = FakeQwenModel()
    qwen_controller = VisualTokenShiftController(model_family="qwen2_5_vl", model_name="fake-qwen")
    qwen_controller.install_qwen_hf_hook(qwen_model)
    qwen_input = torch.tensor([[10.0], [20.0], [30.0], [40.0], [1.0], [2.0], [3.0], [4.0]])
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
    with qwen_controller.sample(
        dataset="sentinel",
        sample_meta={"index": 0},
        topology={"validated": True},
    ):
        qwen_output = qwen_model.visual(qwen_input, grid_thw=grid)
    checks.update(
        {
            "qwen_i1_unchanged": torch.equal(qwen_output.pooler_output[:4], qwen_input[:4]),
            "qwen_i2_1234_to_4123": qwen_output.pooler_output[4:].flatten().tolist()
            == [4.0, 1.0, 2.0, 3.0],
            "qwen_shape_unchanged": qwen_output.pooler_output.shape == qwen_input.shape,
            "qwen_last_hidden_state_unchanged": torch.equal(
                qwen_output.last_hidden_state,
                qwen_input + 1000.0,
            ),
        }
    )

    minicpm_dir = args.output_dir / "minicpm"
    configure(minicpm_dir, "roll_right_1")
    minicpm_input = torch.stack(
        [
            torch.arange(101.0, 165.0).unsqueeze(-1),
            torch.arange(1.0, 65.0).unsqueeze(-1),
        ]
    )
    minicpm_model = FakeMiniCPMModel(minicpm_input)
    minicpm_controller = VisualTokenShiftController(model_family="minicpm45", model_name="fake-minicpm")
    minicpm_controller.install_minicpm_hf_hook(minicpm_model)
    minicpm_data = {
        "image_sizes": [[(448, 448), (448, 448)]],
        "image_bound": [[torch.tensor([0, 64]), torch.tensor([65, 129])]],
        "pixel_values": [[torch.zeros(1), torch.zeros(1)]],
        "tgt_sizes": [[torch.tensor([32, 32]), torch.tensor([32, 32])]],
    }
    base_processor = FakeMiniCPMProcessor(minicpm_data)
    processor_proxy = minicpm_controller.wrap_minicpm_processor(base_processor)
    processor_proxy.tokenizer = "tokenizer-after"
    checks["minicpm_processor_proxy_assignment"] = base_processor.tokenizer == "tokenizer-after"
    with minicpm_controller.sample(
        dataset="sentinel",
        sample_meta={"index": 0},
        topology={"validated": True},
    ):
        captured_data = processor_proxy()
        checks["minicpm_processor_proxy_call"] = captured_data is minicpm_data
        minicpm_data.pop("image_sizes")
        minicpm_output = minicpm_model.get_vision_embedding(minicpm_data)[0]
    checks.update(
        {
            "minicpm_i1_block_unchanged": torch.equal(minicpm_output[0], minicpm_input[0]),
            "minicpm_i2_1_to_64_right_roll": minicpm_output[1].flatten().tolist()
            == [64.0] + list(range(1, 64)),
            "minicpm_shape_unchanged": minicpm_output.shape == minicpm_input.shape,
        }
    )

    raw_files = sorted(args.output_dir.rglob("*.npz"))
    checks["raw_npz_count"] = len(raw_files) == 2
    for raw_path in raw_files:
        with np.load(raw_path) as raw:
            mapping = raw["source_index_for_output"]
            before = raw["target_before"]
            after = raw["target_after"]
            checks[f"raw_exact_{raw_path.parent.name}"] = np.array_equal(after, before[:, mapping, :])
            checks[f"raw_non_target_{raw_path.parent.name}"] = np.array_equal(
                raw["non_target_before"], raw["non_target_after"]
            )
    json_records = []
    for path in sorted(args.output_dir.rglob("visual_token_shift.pid*.jsonl")):
        json_records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    checks["records_apply_once"] = len(json_records) == 2 and all(record["apply_count"] == 1 for record in json_records)
    checks["records_final_container_exact"] = len(json_records) == 2 and all(
        record["final_target_exact"] and record["final_non_target_exact"] for record in json_records
    )

    summary = {"all_passed": all(checks.values()), "checks": checks}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
