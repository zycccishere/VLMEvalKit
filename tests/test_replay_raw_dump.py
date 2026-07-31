import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).parents[1] / "vlmeval" / "vlm" / "replay_policy.py"
SPEC = importlib.util.spec_from_file_location("replay_policy_under_test", MODULE_PATH)
REPLAY_POLICY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REPLAY_POLICY)


class ReplayRawDumpTest(unittest.TestCase):
    def test_raw_dump_records_exact_iqiq_order(self):
        before = [
            {"type": "image", "value": "/tmp/image.png"},
            {"type": "text", "value": "question"},
        ]
        after = REPLAY_POLICY.apply_replay(before, mode="image_text_image_text")

        with tempfile.TemporaryDirectory() as tmpdir:
            dump_path = Path(tmpdir) / "raw" / "replay.jsonl"
            old = os.environ.get("REPLAY_RAW_DUMP_PATH")
            os.environ["REPLAY_RAW_DUMP_PATH"] = str(dump_path)
            try:
                REPLAY_POLICY.maybe_debug_print_replay(
                    enabled=False,
                    mode="image_text_image_text",
                    before=before,
                    after=after,
                    tag="sentinel",
                )
            finally:
                if old is None:
                    os.environ.pop("REPLAY_RAW_DUMP_PATH", None)
                else:
                    os.environ["REPLAY_RAW_DUMP_PATH"] = old

            record = json.loads(dump_path.read_text(encoding="utf-8"))

        self.assertEqual(record["before_counts"], {"text": 1, "image": 1, "video": 0})
        self.assertEqual(record["after_counts"], {"text": 2, "image": 2, "video": 0})
        self.assertEqual([item["type"] for item in record["after_content"]], ["image", "text", "image", "text"])
        self.assertEqual(record["after_content"][0]["value"], record["after_content"][2]["value"])
        self.assertEqual(record["after_content"][1]["value"], record["after_content"][3]["value"])

    def test_validator_rejects_wrong_mode_and_extra_modality(self):
        validator_path = Path(__file__).parents[1] / "scripts" / "validate_replay_raw_dump.py"
        spec = importlib.util.spec_from_file_location("raw_dump_validator", validator_path)
        validator = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(validator)

        record = {
            "mode": "image_text_text",
            "before_counts": {"text": 1, "image": 1, "video": 0},
            "after_counts": {"text": 2, "image": 1, "video": 0},
            "before_content": [
                {"type": "image", "value": "/tmp/missing.png"},
                {"type": "text", "value": "question"},
            ],
            "after_content": [
                {"type": "image", "value": "/tmp/missing.png"},
                {"type": "text", "value": "question"},
                {"type": "audio", "value": "/tmp/audio.wav"},
                {"type": "text", "value": "question"},
            ],
        }
        check = validator.validate_record(record, "iqiq")
        self.assertTrue(any("record mode" in error for error in check["errors"]))
        self.assertTrue(any("after types" in error for error in check["errors"]))

    def test_validator_binds_policy_final_input_and_prediction_identity(self):
        validator_path = Path(__file__).parents[1] / "scripts" / "validate_replay_raw_dump.py"
        spec = importlib.util.spec_from_file_location("raw_dump_validator_bound", validator_path)
        validator = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(validator)

        with tempfile.TemporaryDirectory() as tmpdir:
            from PIL import Image

            image_path = Path(tmpdir) / "image.png"
            Image.new("RGB", (4, 3), (12, 34, 56)).save(image_path)
            source_hash = validator._decoded_rgb_sha256(image_path)
            image_ref = image_path.as_uri()
            policy = {
                "schema_version": 2,
                "stage": "replay_policy",
                "run_uuid": "run-1",
                "matrix": "matrix",
                "task_tag": "task",
                "model_key": "model",
                "dataset": "CountQA",
                "condition": "iq",
                "sample_indices_json": '["sample-1"]',
                "mode": "image_text",
                "before_counts": {"text": 1, "image": 1, "video": 0},
                "after_counts": {"text": 1, "image": 1, "video": 0},
                "before_content": [
                    {"type": "image", "value": image_ref},
                    {"type": "text", "value": "question"},
                ],
                "after_content": [
                    {"type": "image", "value": image_ref},
                    {"type": "text", "value": "question"},
                ],
            }
            final = {
                "schema_version": "final_model_input.v1",
                "stage": "final_model_input",
                "model_family": "qwen2.5-vl",
                "backend": "vllm",
                "batch_position": 0,
                "call_correlation_id": "parent-1:0",
                "parent_call_id": "parent-1",
                "task_identity": {
                    "run_uuid": "run-1",
                    "matrix_tag": "matrix",
                    "task_tag": "task",
                    "model_key": "model",
                    "dataset": "CountQA",
                    "condition": "iq",
                    "canonical_index": "sample-1",
                },
                "missing_task_identity_fields": [],
                "text_chat_representation": {"value": "question"},
                "visual_input_count": 1,
                "visual_inputs": [{
                    "sha256": source_hash,
                    "source_sha256": source_hash,
                    "source_ref": image_ref,
                    "modality": "image",
                    "size": {"width": 4, "height": 3},
                    "unresolved": False,
                }],
                "content_sequence": [
                    {
                        "position": 0,
                        "type": "image",
                        "visual_position": 0,
                    },
                    {
                        "position": 1,
                        "type": "text",
                        "text_sha256": validator.hashlib.sha256(b"question").hexdigest(),
                    },
                ],
                "consumer_api": "vllm.LLM.generate",
            }

            policy_checks, final_checks, global_errors = validator.validate_dump(
                [policy, final],
                condition="iq",
                expected_indices=["sample-1"],
                run_uuid="run-1",
                matrix="matrix",
                task_tag="task",
                model_key="model",
                dataset="CountQA",
                model_family="qwen2.5-vl",
                backend="vllm",
                prediction_indices=["sample-1"],
            )
            self.assertEqual(global_errors, [])
            self.assertEqual(policy_checks[0]["errors"], [])
            self.assertEqual(final_checks[0]["errors"], [])

            final["task_identity"]["model_key"] = "wrong-model"
            _, rejected_final, _ = validator.validate_dump(
                [policy, final],
                condition="iq",
                expected_indices=["sample-1"],
                run_uuid="run-1",
                matrix="matrix",
                task_tag="task",
                model_key="model",
                dataset="CountQA",
                model_family="qwen2.5-vl",
                backend="vllm",
                prediction_indices=["sample-1"],
            )
            self.assertTrue(any("model_key" in error for error in rejected_final[0]["errors"]))

            final["task_identity"]["model_key"] = "model"
            final["visual_inputs"][0]["sha256"] = "wrong-image"
            _, rejected_final, _ = validator.validate_dump(
                [policy, final],
                condition="iq",
                expected_indices=["sample-1"],
                run_uuid="run-1",
                matrix="matrix",
                task_tag="task",
                model_key="model",
                dataset="CountQA",
                model_family="qwen2.5-vl",
                backend="vllm",
                prediction_indices=["sample-1"],
            )
            self.assertTrue(any("content hashes" in error for error in rejected_final[0]["errors"]))

            final["visual_inputs"][0]["sha256"] = source_hash
            final["content_sequence"] = [
                final["content_sequence"][1],
                final["content_sequence"][0],
            ]
            _, rejected_final, _ = validator.validate_dump(
                [policy, final],
                condition="iq",
                expected_indices=["sample-1"],
                run_uuid="run-1",
                matrix="matrix",
                task_tag="task",
                model_key="model",
                dataset="CountQA",
                model_family="qwen2.5-vl",
                backend="vllm",
                prediction_indices=["sample-1"],
            )
            self.assertTrue(any("content sequence" in error for error in rejected_final[0]["errors"]))


if __name__ == "__main__":
    unittest.main()
