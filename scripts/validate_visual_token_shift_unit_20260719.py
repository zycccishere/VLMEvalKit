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
roll_vllm_iqiq_image_embeddings = SHIFT_MODULE.roll_vllm_iqiq_image_embeddings
validate_iqiq_topology = SHIFT_MODULE.validate_iqiq_topology
write_vllm_runtime_contract = SHIFT_MODULE.write_vllm_runtime_contract


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
    cached_pair0 = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    pair1 = torch.tensor([[10.0], [20.0], [30.0]])
    vllm_input = [cached_pair0, cached_pair0, pair1.clone(), pair1.clone()]
    vllm_shifted, vllm_audit, vllm_raw = roll_vllm_iqiq_image_embeddings(
        vllm_input,
        shift=1,
        validate_values=True,
    )
    vllm_noop, vllm_noop_audit, _ = roll_vllm_iqiq_image_embeddings(
        vllm_input,
        shift=0,
        validate_values=True,
    )
    checks.update(
        {
            "vllm_batch_pair_count_two": vllm_audit["request_pair_count"] == 2,
            "vllm_batch_i1_pair0_unchanged": torch.equal(vllm_shifted[0], cached_pair0),
            "vllm_batch_i2_pair0_1234_to_4123": vllm_shifted[1].flatten().tolist()
            == [4.0, 1.0, 2.0, 3.0],
            "vllm_batch_i2_pair1_right_roll": vllm_shifted[3].flatten().tolist()
            == [30.0, 10.0, 20.0],
            "vllm_cache_alias_not_mutated": vllm_input[0].flatten().tolist()
            == [1.0, 2.0, 3.0, 4.0]
            and vllm_input[1].flatten().tolist() == [1.0, 2.0, 3.0, 4.0],
            "vllm_shift_out_of_place": vllm_shifted[1].data_ptr() != vllm_input[1].data_ptr(),
            "vllm_raw_has_both_pairs": len(vllm_raw) == 2,
            "vllm_noop_values_exact": all(
                torch.equal(before, after)
                for before, after in zip(vllm_input, vllm_noop)
            ),
            "vllm_noop_uses_same_audit_path": all(
                pair_record["shift"] == 0 and pair_record["i2_roll_exact"]
                for pair_record in vllm_noop_audit["pair_records"]
            ),
        }
    )
    try:
        roll_vllm_iqiq_image_embeddings(vllm_input[:3], shift=1)
    except ValueError:
        checks["vllm_odd_item_count_fails_closed"] = True
    else:
        checks["vllm_odd_item_count_fails_closed"] = False
    unequal = [cached_pair0, cached_pair0 + 1]
    try:
        roll_vllm_iqiq_image_embeddings(unequal, shift=1, validate_values=True)
    except ValueError:
        checks["vllm_nonduplicate_pair_fails_closed"] = True
    else:
        checks["vllm_nonduplicate_pair_fails_closed"] = False
    _, deferred_audit, _ = roll_vllm_iqiq_image_embeddings(
        [cached_pair0.clone(), cached_pair0.clone(), pair1.clone(), pair1.clone()],
        shift=1,
        require_pair_equality=True,
    )
    checks["vllm_strict_batch_equality_aggregated"] = all(
        pair_record["i1_i2_equal_exact"] is True
        for pair_record in deferred_audit["pair_records"]
    )
    try:
        roll_vllm_iqiq_image_embeddings(
            unequal,
            shift=1,
            require_pair_equality=True,
        )
    except ValueError:
        checks["vllm_strict_deferred_nonduplicate_fails_closed"] = True
    else:
        checks["vllm_strict_deferred_nonduplicate_fails_closed"] = False
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

    contract_dir = args.output_dir / "runtime-contract"
    configure(contract_dir, "roll_right_1")
    os.environ["REPLAY_VLLM_RUNTIME_CONTRACT"] = "1"
    os.environ["REPLAY_VISUAL_TOKEN_SHIFT_RUN_ID"] = "unit-runtime-contract"
    write_vllm_runtime_contract(
        model_family="qwen2_5_vl",
        dataset="sentinel",
        requests=[
            {"prompt": "first", "image": torch.tensor([1, 2])},
            {"prompt": "second", "image": torch.tensor([3, 4])},
        ],
        sampling_params=SimpleNamespace(
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            stop_token_ids=[1, 2],
            seed=0,
        ),
    )
    contract_path = next(contract_dir.glob("vllm_runtime_contract.pid*.jsonl"))
    contract = json.loads(contract_path.read_text().strip())
    checks["runtime_contract_two_distinct_requests"] = (
        contract["request_count"] == 2
        and len(set(contract["request_hashes"])) == 2
    )
    checks["runtime_contract_sampling_exact"] = contract["sampling_contract"] == {
        "max_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "stop_token_ids": [1, 2],
        "seed": 0,
    }
    os.environ["REPLAY_VLLM_RUNTIME_CONTRACT_LEVEL"] = "count"
    write_vllm_runtime_contract(
        model_family="qwen2_5_vl",
        dataset="sentinel",
        requests=[object(), object()],
        sampling_params=SimpleNamespace(max_tokens=128),
    )
    count_contract = json.loads(contract_path.read_text().splitlines()[-1])
    checks["runtime_count_contract_skips_request_serialization"] = (
        count_contract["request_identity_level"] == "count"
        and count_contract["request_count"] == 2
        and count_contract["request_hashes"] == []
    )
    os.environ["REPLAY_VLLM_RUNTIME_CONTRACT_LEVEL"] = "full"
    os.environ["REPLAY_VLLM_RUNTIME_CONTRACT"] = "0"

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
