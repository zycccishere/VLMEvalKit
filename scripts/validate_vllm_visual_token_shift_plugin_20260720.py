#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class FakeMultiModalRegistry:
    def register_processor(self, *args, **kwargs):
        def decorate(model_cls):
            return model_cls

        return decorate


class FakeVLLMBase:
    def __init__(self, *, vllm_config, prefix=""):
        self.captured_embeddings = None
        self.visual = SimpleNamespace(spatial_merge_size=1)
        self.config = SimpleNamespace(
            image_token_id=99,
            _name_or_path="synthetic-vllm-model",
            query_num=1,
        )

    def _scatter(self, input_ids, multimodal_embeddings, is_multimodal):
        output = input_ids.float().unsqueeze(-1)
        if multimodal_embeddings:
            output[is_multimodal] = torch.cat(list(multimodal_embeddings), dim=0)
        return output

    def embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
        handle_oov_mm_token=False,
    ):
        self.captured_embeddings = multimodal_embeddings
        return self._scatter(input_ids, multimodal_embeddings, is_multimodal)

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None):
        self.captured_embeddings = multimodal_embeddings
        return self._scatter(input_ids, multimodal_embeddings, input_ids == 99)

    def _process_image_input(self, image_input):
        if isinstance(image_input, dict) and "embeddings" in image_input:
            return image_input["embeddings"]
        return image_input

    def _process_vision_input(self, image_input):
        return image_input["embeddings"]

    def get_input_embeddings_v0(self, input_ids, image_input=None, video_input=None):
        image_embeddings = self._process_image_input(image_input)
        self.captured_embeddings = image_embeddings
        return self._scatter(input_ids, image_embeddings, input_ids == 99)


class FakeModelRegistry:
    registrations = {}

    @classmethod
    def register_model(cls, architecture, target):
        cls.registrations[architecture] = target


def install_fake_vllm() -> None:
    vllm = ModuleType("vllm")
    vllm.ModelRegistry = FakeModelRegistry
    multimodal = ModuleType("vllm.multimodal")
    multimodal.MULTIMODAL_REGISTRY = FakeMultiModalRegistry()
    model_executor = ModuleType("vllm.model_executor")
    models = ModuleType("vllm.model_executor.models")
    qwen = ModuleType("vllm.model_executor.models.qwen2_5_vl")
    qwen.Qwen2_5_VLDummyInputsBuilder = object
    qwen.Qwen2_5_VLForConditionalGeneration = FakeVLLMBase
    qwen.Qwen2_5_VLMultiModalProcessor = object
    qwen.Qwen2_5_VLProcessingInfo = object
    minicpmo = ModuleType("vllm.model_executor.models.minicpmo")
    minicpmo.MiniCPMO4_5 = FakeVLLMBase
    minicpmo.MiniCPMODummyInputsBuilder = object
    minicpmo.MiniCPMOMultiModalProcessor = object
    minicpmo.MiniCPMOProcessingInfo = object
    sys.modules.update(
        {
            "vllm": vllm,
            "vllm.multimodal": multimodal,
            "vllm.model_executor": model_executor,
            "vllm.model_executor.models": models,
            "vllm.model_executor.models.qwen2_5_vl": qwen,
            "vllm.model_executor.models.minicpmo": minicpmo,
        }
    )


def install_vlmeval_test_namespace() -> None:
    vlmeval = ModuleType("vlmeval")
    vlmeval.__path__ = [str(REPO_ROOT / "vlmeval")]
    vlm = ModuleType("vlmeval.vlm")
    vlm.__path__ = [str(REPO_ROOT / "vlmeval" / "vlm")]
    sys.modules.update({"vlmeval": vlmeval, "vlmeval.vlm": vlm})


def configure(dump_dir: Path) -> None:
    os.environ.update(
        {
            "REPLAY_VISUAL_TOKEN_SHIFT": "roll_right_1",
            "REPLAY_VISUAL_TOKEN_SHIFT_TARGET_POSITION": "2",
            "REPLAY_VISUAL_TOKEN_SHIFT_CHUNKED_PREFILL_DISABLED": "1",
            "REPLAY_VISUAL_TOKEN_SHIFT_PREFIX_CACHING_DISABLED": "1",
            "REPLAY_VISUAL_TOKEN_SHIFT_DUMP_DIR": str(dump_dir),
            "REPLAY_VISUAL_TOKEN_SHIFT_DUMP_SAMPLES": "1",
            "REPLAY_VISUAL_TOKEN_SHIFT_RAW_DUMP": "1",
            "REPLAY_VISUAL_TOKEN_SHIFT_FULL_VALIDATION": "1",
            "REPLAY_MODE": "image_text_image_text",
            "REPLAY_TIMES": "1",
            "REPLAY_IMAGE_TRANSFORM": "baseline",
            "REPLAY_VISUAL_TOKEN_SHIFT_RUN_ID": "synthetic-three-request-batch",
            "REPLAY_VLLM_TARGET_FAMILY": "qwen2_5_vl",
            "REPLAY_MINICPM_EXPECTED_QUERY_NUM": "1",
            "VLLM_USE_V1": "1",
        }
    )


def pair(values):
    tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)
    return tensor, tensor


def input_ids_for_items(items):
    values = [7]
    for item in items:
        values.extend([99] * int(item.shape[0]))
        values.append(8)
    return torch.tensor(values)


def input_ids_with_split_spans(items):
    values = [7]
    for item in items:
        for token_index in range(int(item.shape[0])):
            values.append(99)
            if token_index + 1 < int(item.shape[0]):
                values.append(8)
        values.append(7)
    return torch.tensor(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure(args.output_dir)
    install_fake_vllm()
    install_vlmeval_test_namespace()

    models = importlib.import_module("vlmeval.vlm.replay_vllm_visual_token_models")
    plugin = importlib.import_module("vlmeval.vlm.replay_vllm_visual_token_plugin")
    plugin.register()
    checks = {
        "plugin_registered_qwen": "Qwen2_5_VLForConditionalGeneration"
        in FakeModelRegistry.registrations,
        "qwen_plugin_did_not_register_minicpm": "MiniCPMO"
        not in FakeModelRegistry.registrations,
    }
    for name, model_cls in (
        ("qwen", models.ReplayShiftQwen2_5VL),
        ("minicpm", models.ReplayShiftMiniCPMO45),
    ):
        signature = inspect.signature(model_cls.__init__)
        checks[f"{name}_loader_signature_has_vllm_config"] = (
            "vllm_config" in signature.parameters
            and signature.parameters["vllm_config"].kind
            is inspect.Parameter.KEYWORD_ONLY
        )
        checks[f"{name}_loader_signature_has_prefix"] = (
            "prefix" in signature.parameters
            and signature.parameters["prefix"].kind
            is inspect.Parameter.KEYWORD_ONLY
        )
    os.environ["REPLAY_VLLM_TARGET_FAMILY"] = "minicpm_o_4_5"
    plugin.register()
    checks["plugin_registered_minicpm"] = "MiniCPMO" in FakeModelRegistry.registrations

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=False,
            disable_chunked_mm_input=True,
            max_num_seqs=2,
            max_num_batched_tokens=32768,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )
    for check_name, bad_config, expected_text in (
        (
            "rejects_chunked_prefill",
            SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    enable_chunked_prefill=True,
                    disable_chunked_mm_input=True,
                ),
                cache_config=SimpleNamespace(enable_prefix_caching=False),
            ),
            "enable_chunked_prefill=False",
        ),
        (
            "requires_disable_chunked_mm_input",
            SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    enable_chunked_prefill=False,
                    disable_chunked_mm_input=False,
                ),
                cache_config=SimpleNamespace(enable_prefix_caching=False),
            ),
            "disable_chunked_mm_input=True",
        ),
        (
            "rejects_prefix_caching",
            SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    enable_chunked_prefill=False,
                    disable_chunked_mm_input=True,
                ),
                cache_config=SimpleNamespace(enable_prefix_caching=True),
            ),
            "enable_prefix_caching=False",
        ),
        (
            "rejects_unknown_chunked_prefill",
            SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    enable_chunked_prefill=None,
                    disable_chunked_mm_input=True,
                ),
                cache_config=SimpleNamespace(enable_prefix_caching=False),
            ),
            "enable_chunked_prefill=False",
        ),
        (
            "rejects_unknown_prefix_caching",
            SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    enable_chunked_prefill=False,
                    disable_chunked_mm_input=True,
                ),
                cache_config=SimpleNamespace(enable_prefix_caching=None),
            ),
            "enable_prefix_caching=False",
        ),
    ):
        try:
            models.ReplayShiftQwen2_5VL(vllm_config=bad_config)
        except RuntimeError as exc:
            checks[check_name] = expected_text in str(exc)
        else:
            checks[check_name] = False
    os.environ["REPLAY_VLLM_TARGET_FAMILY"] = "qwen2_5_vl"
    qwen = models.ReplayShiftQwen2_5VL(vllm_config=config)
    shifts = importlib.import_module("vlmeval.vlm.replay_visual_token_shift")
    checks["normalizes_vllm_v0_multimodal_limit"] = (
        shifts._extract_limit_mm_per_prompt(
            SimpleNamespace(
                get_multimodal_config=lambda: SimpleNamespace(
                    limit_per_prompt={"image": 8}
                )
            )
        )
        == {"image": 8}
    )
    checks["normalizes_vllm_v1_multimodal_limit"] = (
        shifts._extract_limit_mm_per_prompt(
            SimpleNamespace(
                multimodal_config=SimpleNamespace(
                    limit_per_prompt={
                        "image": SimpleNamespace(count=8),
                    }
                )
            )
        )
        == {"image": 8}
    )
    handshakes = shifts.require_vllm_visual_token_shift_worker_handshake(
        model_family="qwen2_5_vl",
        timeout_seconds=0,
    )
    checks["qwen_worker_handshake_found"] = len(handshakes) == 1
    p0 = pair([1, 2, 3, 4])
    p1 = pair([10, 20, 30])
    p2 = pair([100, 200])
    original = [*p0, *p1, *p2]
    input_ids = input_ids_for_items(original)
    qwen.get_input_embeddings(input_ids, original)
    checks["profiling_roll_applied_without_recording"] = (
        qwen.captured_embeddings[1].flatten().tolist() == [4, 1, 2, 3]
        and qwen._replay_shift_call_count == 0
        and not list(args.output_dir.glob("visual_token_shift_calls.vllm.pid*.jsonl"))
    )
    shifts.arm_vllm_visual_token_shift_recording(model_family="qwen2_5_vl")
    qwen.get_input_embeddings(input_ids, original)
    qwen.get_input_embeddings(input_ids, original)
    shifted = qwen.captured_embeddings
    checks.update(
        {
            "qwen_v1_pair0_i1_unchanged": shifted[0].flatten().tolist() == [1, 2, 3, 4],
            "qwen_v1_pair0_i2_rolled": shifted[1].flatten().tolist() == [4, 1, 2, 3],
            "qwen_v1_pair1_i1_unchanged": shifted[2].flatten().tolist() == [10, 20, 30],
            "qwen_v1_pair1_i2_rolled": shifted[3].flatten().tolist() == [30, 10, 20],
            "qwen_v1_pair2_i1_unchanged": shifted[4].flatten().tolist() == [100, 200],
            "qwen_v1_pair2_i2_rolled": shifted[5].flatten().tolist() == [200, 100],
            "qwen_cache_source_not_mutated": original[1].flatten().tolist() == [1, 2, 3, 4],
        }
    )

    os.environ["REPLAY_VLLM_TARGET_FAMILY"] = "minicpm_o_4_5"
    minicpm = models.ReplayShiftMiniCPMO45(vllm_config=config)
    minicpm_items = [
        *pair([5, 6, 7]),
        *pair([11, 12]),
        *pair([21, 22, 23, 24]),
    ]
    minicpm._process_vision_input(
        {
            "type": "pixel_values",
            "num_slices": torch.tensor(
                [int(item.shape[0]) for item in minicpm_items],
                dtype=torch.long,
            ),
            "embeddings": minicpm_items,
        }
    )
    shifts.arm_vllm_visual_token_shift_recording(model_family="minicpm_o_4_5")
    minicpm_input_ids = input_ids_with_split_spans(minicpm_items)
    minicpm_is_multimodal = minicpm_input_ids == 99
    minicpm.embed_input_ids(
        minicpm_input_ids,
        minicpm_items,
        is_multimodal=minicpm_is_multimodal,
    )
    checks.update(
        {
            "minicpm_whole_series_roll_pair0": minicpm.captured_embeddings[1].flatten().tolist()
            == [7, 5, 6],
            "minicpm_whole_series_roll_pair1": minicpm.captured_embeddings[3].flatten().tolist()
            == [12, 11],
            "minicpm_whole_series_roll_pair2": minicpm.captured_embeddings[5].flatten().tolist()
            == [24, 21, 22, 23],
        }
    )

    os.environ["VLLM_USE_V1"] = "0"
    os.environ["REPLAY_VLLM_TARGET_FAMILY"] = "qwen2_5_vl"
    legacy = models.ReplayShiftQwen2_5VL(vllm_config=config)
    legacy_items = [
        *pair([1, 2, 3, 4]),
        *pair([10, 20, 30]),
        *pair([100, 200]),
    ]
    legacy_input_ids = input_ids_for_items(legacy_items)
    legacy_image_input = {
        "embeddings": legacy_items,
        "image_grid_thw": torch.tensor(
            [[1, 1, int(item.shape[0])] for item in legacy_items]
        ),
    }
    legacy.get_input_embeddings_v0(
        legacy_input_ids,
        image_input=legacy_image_input,
    )
    checks["qwen_legacy_v0_post_projector_roll"] = (
        legacy.captured_embeddings[1].flatten().tolist() == [4, 1, 2, 3]
        and legacy.captured_embeddings[3].flatten().tolist() == [30, 10, 20]
        and legacy.captured_embeddings[5].flatten().tolist() == [200, 100]
    )

    os.environ["VLLM_USE_V1"] = "1"
    os.environ["REPLAY_VISUAL_TOKEN_SHIFT"] = "noop_vllm"
    os.environ["REPLAY_VLLM_TARGET_FAMILY"] = "qwen2_5_vl"
    noop = models.ReplayShiftQwen2_5VL(vllm_config=config)
    noop.get_input_embeddings(input_ids, original)
    checks["qwen_matched_runtime_noop_exact"] = all(
        torch.equal(before, after)
        for before, after in zip(original, noop.captured_embeddings)
    )

    os.environ["REPLAY_VLLM_TARGET_FAMILY"] = "minicpm_o_4_5"
    minicpm_noop = models.ReplayShiftMiniCPMO45(vllm_config=config)
    minicpm_noop._process_vision_input(
        {
            "type": "pixel_values",
            "num_slices": torch.tensor(
                [int(item.shape[0]) for item in minicpm_items],
                dtype=torch.long,
            ),
            "embeddings": minicpm_items,
        }
    )
    minicpm_noop.embed_input_ids(
        minicpm_input_ids,
        minicpm_items,
        is_multimodal=minicpm_is_multimodal,
    )
    checks["minicpm_matched_runtime_noop_exact"] = all(
        torch.equal(before, after)
        for before, after in zip(minicpm_items, minicpm_noop.captured_embeddings)
    )

    os.environ["VLLM_USE_V1"] = "0"
    os.environ["REPLAY_VLLM_TARGET_FAMILY"] = "qwen2_5_vl"
    legacy_noop = models.ReplayShiftQwen2_5VL(vllm_config=config)
    legacy_noop.get_input_embeddings_v0(
        legacy_input_ids,
        image_input=legacy_image_input,
    )
    checks["qwen_legacy_v0_matched_runtime_noop_exact"] = all(
        torch.equal(before, after)
        for before, after in zip(legacy_items, legacy_noop.captured_embeddings)
    )

    light_dir = args.output_dir / "lightweight"
    light_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "VLLM_USE_V1": "1",
            "REPLAY_VISUAL_TOKEN_SHIFT": "roll_right_1",
            "REPLAY_VLLM_TARGET_FAMILY": "qwen2_5_vl",
            "REPLAY_VISUAL_TOKEN_SHIFT_RUN_ID": "synthetic-lightweight",
            "REPLAY_VISUAL_TOKEN_SHIFT_DUMP_DIR": str(light_dir),
            "REPLAY_VISUAL_TOKEN_SHIFT_FULL_VALIDATION": "0",
        }
    )
    light = models.ReplayShiftQwen2_5VL(vllm_config=config)
    light.get_input_embeddings(input_ids, original)
    shifts.arm_vllm_visual_token_shift_recording(model_family="qwen2_5_vl")
    light.get_input_embeddings(input_ids, original)
    light_calls = [
        json.loads(line)
        for path in light_dir.glob("visual_token_shift_calls.vllm.pid*.jsonl")
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    light_records = [
        json.loads(line)
        for path in light_dir.glob("visual_token_shift.vllm.pid*.jsonl")
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    checks["lightweight_roll_exact_without_tensor_audit"] = (
        light.captured_embeddings[1].flatten().tolist() == [4, 1, 2, 3]
        and len(light_calls) == 1
        and len(light_records) == 1
        and light_calls[0]["validation_level"] == "lightweight"
        and light_calls[0]["full_tensor_validation"] is False
        and light_calls[0]["output_shape_matches_input_ids"] is True
        and light_calls[0]["all_iqiq_pairs_equal_exact"] is None
        and light_calls[0]["pair_structure_validated"] is True
        and light_records[0]["audit"]["pair_equality_required"] is False
        and light_records[0]["final_mm_scatter_exact"] is None
        and not list(light_dir.glob("*.npz"))
    )

    records = []
    for path in args.output_dir.glob("visual_token_shift.vllm.pid*.jsonl"):
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    checks["six_runtime_records"] = len(records) == 6
    checks["records_input_and_mask_unchanged"] = all(
        record["input_ids_unchanged_exact"] and record["is_multimodal_unchanged_exact"]
        for record in records
    )
    checks["records_final_scatter_exact"] = all(
        record["final_mm_scatter_exact"] is True for record in records
    )
    checks["records_are_armed_real_requests"] = all(
        record["real_request"] is True and record["recording_armed"] is True
        for record in records
    )
    minicpm_records = [
        record for record in records if record["model_family"] == "minicpm_o_4_5"
    ]
    checks["minicpm_split_placeholder_spans_accepted"] = bool(minicpm_records) and all(
        record["item_token_coverage_exact"] is True
        and record["item_span_grouping_exact"] is True
        and record["item_span_lengths_required"] is False
        and record["item_span_lengths_exact"] is None
        and record["minicpm_placeholder_token_contract_exact"] is True
        and record["minicpm_placeholder_slice_counts"]
        == [len(group) for group in record["item_span_groups"]]
        and all(
            span_length == 1
            for group in record["item_span_groups"]
            for span_length in group
        )
        and len(record["multimodal_span_lengths"]) > len(record["item_token_counts"])
        for record in minicpm_records
    )
    call_records = []
    for path in args.output_dir.glob("visual_token_shift_calls.vllm.pid*.jsonl"):
        call_records.extend(
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    checks["seven_all_prefill_call_records"] = len(call_records) == 7
    checks["all_prefill_calls_exact"] = all(
        record["all_iqiq_pairs_equal_exact"]
        and record["input_ids_unchanged_exact"]
        and record["is_multimodal_unchanged_exact"]
        and record["final_mm_scatter_exact"]
        and record["item_token_coverage_exact"]
        and record["item_span_grouping_exact"]
        and (
            not record["item_span_lengths_required"]
            or record["item_span_lengths_exact"]
        )
        and record["real_request"]
        and record["recording_armed"]
        for record in call_records
    )
    qwen_roll_calls = sorted(
        record["call_index"]
        for record in call_records
        if record["model_family"] == "qwen2_5_vl"
        and record["mode"] == "roll_right_1"
        and record["stage"] == "post_gather_pre_llm_get_input_embeddings"
    )
    checks["qwen_all_prefill_call_indices_contiguous"] = qwen_roll_calls == [1, 2]
    raw_paths = sorted(args.output_dir.glob("*.npz"))
    checks["six_raw_npz_dumps"] = len(raw_paths) == 6
    for raw_path in raw_paths:
        with np.load(raw_path) as raw:
            pair_indices = sorted(
                int(key.split("_")[0].removeprefix("pair"))
                for key in raw.files
                if key.endswith("_i2_before")
            )
            for pair_index in pair_indices:
                before = raw[f"pair{pair_index}_i2_before"]
                after = raw[f"pair{pair_index}_i2_after"]
                source = raw[f"pair{pair_index}_source_index_for_output"].astype(int)
                checks[f"raw_exact_{raw_path.stem}_pair{pair_index}"] = np.array_equal(
                    after,
                    before[source],
                )

    fake_v0_engine = type("LLMEngine", (), {})
    fake_v0_engine.__module__ = "vllm.engine.llm_engine"
    fake_v1_engine = type("LLMEngine", (), {})
    fake_v1_engine.__module__ = "vllm.v1.engine.llm_engine"
    checks["actual_v0_engine_classified"] = shifts.record_vllm_engine_identity(
        SimpleNamespace(engine_class=fake_v0_engine)
    )["mode"] == "v0"
    checks["actual_v1_engine_classified"] = shifts.record_vllm_engine_identity(
        SimpleNamespace(engine_class=fake_v1_engine)
    )["mode"] == "v1"

    os.environ.update(
        {
            "REPLAY_VISUAL_TOKEN_SHIFT": "none",
            "REPLAY_VLLM_MATCHED_TOKEN_ROLL_RUNTIME": "1",
            "REPLAY_VLLM_TARGET_FAMILY": "qwen2_5_vl",
            "REPLAY_VISUAL_TOKEN_SHIFT_RUN_ID": "synthetic-matched-none",
            "REPLAY_INFERENCE_FINGERPRINT": "synthetic-none-fingerprint",
            "VLLM_USE_V1": "0",
        }
    )
    plugin.register()
    checks["matched_none_plugin_registered"] = (
        "Qwen2_5_VLForConditionalGeneration" in FakeModelRegistry.registrations
    )
    models.ReplayShiftQwen2_5VL(vllm_config=config)
    none_handshakes = shifts.require_vllm_visual_token_shift_worker_handshake(
        model_family="qwen2_5_vl",
        timeout_seconds=0,
    )
    checks["matched_none_worker_handshake_found"] = (
        len(none_handshakes) == 1
        and none_handshakes[0]["mode"] == "none"
        and none_handshakes[0]["scheduler_enable_chunked_prefill"] is False
        and none_handshakes[0]["cache_enable_prefix_caching"] is False
    )

    summary = {"all_passed": all(checks.values()), "checks": checks}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
