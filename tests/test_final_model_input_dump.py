import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch


MODULE_PATH = Path(__file__).parents[1] / "vlmeval" / "vlm" / "final_model_input_dump.py"
SPEC = importlib.util.spec_from_file_location("final_model_input_dump_under_test", MODULE_PATH)
FINAL_DUMP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FINAL_DUMP)

IDENTITY_ENV_NAMES = sorted(
    {name for names in FINAL_DUMP._IDENTITY_ENV_NAMES.values() for name in names}
)


def _write_records_in_process(path, process_index, count):
    os.environ["REPLAY_RAW_DUMP_PATH"] = str(path)
    for offset in range(count):
        FINAL_DUMP.dump_final_model_input(
            model_family="sentinel",
            backend="unit",
            consumer_api="fake.generate",
            text_chat_representation=f"process={process_index};offset={offset}",
            visual_inputs=[],
            call_id=f"{process_index}:{offset}",
        )


class FinalModelInputDumpTest(unittest.TestCase):
    def setUp(self):
        self.old_env = {
            name: os.environ.get(name)
            for name in ["REPLAY_RAW_DUMP_PATH", *IDENTITY_ENV_NAMES]
        }
        for name in self.old_env:
            os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self.old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_dump_is_opt_in(self):
        payload = FINAL_DUMP.dump_final_model_input(
            model_family="qwen2.5-vl",
            backend="vllm",
            consumer_api="fake.generate",
            text_chat_representation="prompt",
            visual_inputs=[],
        )
        self.assertIsNone(payload)

    def test_schema_visual_order_hash_and_identity_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dump_path = Path(tmpdir) / "nested" / "raw.jsonl"
            os.environ["REPLAY_RAW_DUMP_PATH"] = str(dump_path)
            os.environ["REPLAY_RUN_UUID"] = "run-sentinel"
            os.environ["MATRIX_TAG"] = "matrix-sentinel"
            os.environ["TASK_TAG"] = "task-sentinel"
            os.environ["MODEL_KEY"] = "model-from-env"
            os.environ["REPLAY_MODE"] = "image_text_image_text"

            first = Image.new("RGB", (3, 2), (255, 0, 0))
            second = Image.new("RGB", (5, 4), (0, 0, 255))
            payload = FINAL_DUMP.dump_final_model_input(
                model_family="qwen2.5-vl",
                backend="vllm",
                consumer_api="vllm.LLM.generate",
                text_chat_representation={"kind": "prompt", "value": "<image> question"},
                visual_inputs=[
                    FINAL_DUMP.visual_spec(first, source_ref="first.png"),
                    FINAL_DUMP.visual_spec(second, source_ref="second.png"),
                ],
                content_sequence=FINAL_DUMP.summarize_content_sequence(
                    [
                        {"type": "image", "value": "first.png"},
                        {"type": "text", "value": "question"},
                        {"type": "image", "value": "second.png"},
                        {"type": "text", "value": "question"},
                    ]
                ),
                processor_inputs={"input_ids": torch.tensor([[1, 2, 3]])},
                dataset="CountQA",
                model_key="fallback-model-key",
                sample_meta={"sample_index": "countqa-0001-00"},
                call_id="fixed-call-id",
            )

            record = json.loads(dump_path.read_text(encoding="utf-8"))

        self.assertEqual(record, payload)
        self.assertEqual(record["schema_version"], "final_model_input.v1")
        self.assertEqual(record["stage"], "final_model_input")
        self.assertEqual(record["call_correlation_id"], "fixed-call-id")
        self.assertEqual(record["visual_input_count"], 2)
        self.assertEqual([item["position"] for item in record["visual_inputs"]], [0, 1])
        self.assertEqual(record["visual_inputs"][0]["size"], {"width": 3, "height": 2})
        self.assertEqual(record["visual_inputs"][1]["size"], {"width": 5, "height": 4})
        self.assertNotEqual(
            record["visual_inputs"][0]["sha256"],
            record["visual_inputs"][1]["sha256"],
        )
        self.assertEqual(record["task_identity"]["matrix_tag"], "matrix-sentinel")
        self.assertEqual(record["task_identity"]["run_uuid"], "run-sentinel")
        self.assertEqual(record["task_identity"]["task_tag"], "task-sentinel")
        self.assertEqual(record["task_identity"]["model_key"], "model-from-env")
        self.assertEqual(record["task_identity"]["dataset"], "CountQA")
        self.assertEqual(record["task_identity"]["condition"], "image_text_image_text")
        self.assertEqual(record["task_identity"]["canonical_index"], "countqa-0001-00")
        self.assertEqual(record["task_identity_sources"]["model_key"], "env:MODEL_KEY")
        self.assertEqual(
            record["task_identity_sources"]["canonical_index"],
            "replay_meta:sample_index",
        )
        self.assertEqual(record["missing_task_identity_fields"], [])
        self.assertEqual(
            [item["type"] for item in record["content_sequence"]],
            ["image", "text", "image", "text"],
        )
        self.assertEqual(
            record["processor_input_summary"]["input_ids"]["shape"],
            [1, 3],
        )

    def test_missing_index_is_explicit_and_call_id_remains_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["REPLAY_RAW_DUMP_PATH"] = str(Path(tmpdir) / "raw.jsonl")
            payload = FINAL_DUMP.dump_final_model_input(
                model_family="gemma3",
                backend="vllm",
                consumer_api="fake.generate",
                text_chat_representation="prompt",
                visual_inputs=[],
                call_id="correlation-sentinel",
            )

        self.assertIsNotNone(payload)
        self.assertIsNone(payload["task_identity"]["canonical_index"])
        self.assertIn("canonical_index", payload["missing_task_identity_fields"])
        self.assertEqual(payload["call_correlation_id"], "correlation-sentinel")

    def test_multiprocess_append_produces_complete_json_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dump_path = Path(tmpdir) / "raw.jsonl"
            context = multiprocessing.get_context("fork")
            processes = [
                context.Process(target=_write_records_in_process, args=(dump_path, process_index, 20))
                for process_index in range(4)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=20)
                self.assertEqual(process.exitcode, 0)

            records = [json.loads(line) for line in dump_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(records), 80)
        self.assertEqual(len({record["call_correlation_id"] for record in records}), 80)
        self.assertTrue(all(record["stage"] == "final_model_input" for record in records))

    def test_observe_bound_method_sees_final_arguments_and_restores_method(self):
        class Owner:
            def generate(self, prompt, *, scale=1):
                return prompt * scale

        owner = Owner()
        original = owner.generate
        observed = []

        with FINAL_DUMP.observe_bound_method(
            owner,
            "generate",
            lambda args, kwargs: observed.append((args, kwargs)),
        ):
            self.assertEqual(owner.generate("x", scale=3), "xxx")

        self.assertEqual(observed, [(('x',), {"scale": 3})])
        self.assertEqual(owner.generate("y", scale=2), "yy")
        self.assertEqual(owner.generate.__func__, original.__func__)

    def test_observe_bound_method_restores_after_exception(self):
        class Owner:
            def generate(self):
                raise RuntimeError("sentinel")

        owner = Owner()
        original = owner.generate
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with FINAL_DUMP.observe_bound_method(owner, "generate", lambda args, kwargs: None):
                owner.generate()
        self.assertEqual(owner.generate.__func__, original.__func__)


if __name__ == "__main__":
    unittest.main()
