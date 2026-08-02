import os
import unittest
from argparse import Namespace

_IMPORT_ENV_KEYS = (
    "VLMEVAL_USE_QWEN_MINIMAL_CONFIG",
    "VLMEVAL_API_MINIMAL_IMPORT",
    "VLMEVAL_VLM_MINIMAL_IMPORT",
    "VLMEVAL_LAZY_INIT",
)
_IMPORT_ENV = {key: os.environ.get(key) for key in _IMPORT_ENV_KEYS}
os.environ.setdefault("VLMEVAL_USE_QWEN_MINIMAL_CONFIG", "1")
os.environ.setdefault("VLMEVAL_API_MINIMAL_IMPORT", "1")
os.environ.setdefault("VLMEVAL_VLM_MINIMAL_IMPORT", "1")
os.environ.setdefault("VLMEVAL_LAZY_INIT", "1")

from run import (
    QWEN25VL_DATASET_RUNTIME_KEYS,
    _is_qwen25vl_model,
    apply_dataset_runtime_policy,
    capture_dataset_runtime_baseline,
)

for key, value in _IMPORT_ENV.items():
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


class DatasetRuntimePolicyTest(unittest.TestCase):
    def setUp(self):
        self.saved_env = {key: os.environ.get(key) for key in QWEN25VL_DATASET_RUNTIME_KEYS}
        for key in QWEN25VL_DATASET_RUNTIME_KEYS:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self.saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_non_logicvista_preserves_explicit_sampling_override(self):
        os.environ["QWEN2VL_VLLM_TEMPERATURE"] = "0"
        os.environ["QWEN2VL_VLLM_TOP_P"] = "1"
        os.environ["QWEN2VL_VLLM_MAX_TOKENS"] = "32"
        baseline = capture_dataset_runtime_baseline(Namespace(batch_size=64))

        policy = apply_dataset_runtime_policy(
            Namespace(batch_size=64),
            "Qwen2.5-VL-3B-Instruct-Replay",
            "CountQA",
            runtime_baseline=baseline,
        )

        self.assertEqual(policy[0], "default")
        self.assertEqual(os.environ["VLLM_USE_V1"], "1")
        self.assertEqual(os.environ["QWEN2VL_VLLM_TEMPERATURE"], "0")
        self.assertEqual(os.environ["QWEN2VL_VLLM_TOP_P"], "1")
        self.assertEqual(os.environ["QWEN2VL_VLLM_MAX_TOKENS"], "32")

    def test_leaving_logicvista_restores_explicit_sampling_override(self):
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_MAX_NUM_SEQS"] = "17"
        os.environ["QWEN2VL_VLLM_MAX_TOKENS"] = "32"
        os.environ["QWEN2VL_VLLM_STOP_TOKEN_IDS"] = "42,43"
        args = Namespace(batch_size=64)
        baseline = capture_dataset_runtime_baseline(args)

        apply_dataset_runtime_policy(
            args,
            "Qwen2.5-VL-3B-Instruct-Replay",
            "LogicVista",
            runtime_baseline=baseline,
        )
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(os.environ["VLLM_USE_V1"], "0")
        self.assertEqual(os.environ["VLLM_MAX_NUM_SEQS"], "128")
        self.assertEqual(os.environ["QWEN2VL_VLLM_MAX_TOKENS"], "32")
        self.assertEqual(os.environ["QWEN2VL_VLLM_STOP_TOKEN_IDS"], "42,43")

        apply_dataset_runtime_policy(
            args,
            "Qwen2.5-VL-3B-Instruct-Replay",
            "CountQA",
            runtime_baseline=baseline,
        )
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(os.environ["VLLM_USE_V1"], "1")
        self.assertEqual(os.environ["VLLM_MAX_NUM_SEQS"], "17")
        self.assertEqual(os.environ["QWEN2VL_VLLM_MAX_TOKENS"], "32")
        self.assertEqual(os.environ["QWEN2VL_VLLM_STOP_TOKEN_IDS"], "42,43")
        self.assertNotIn("LOGICVISTA_QWEN25VL_LEGACY_SAMPLING", os.environ)

    def test_leaving_logicvista_drops_only_policy_generated_sampling(self):
        args = Namespace(batch_size=64)
        baseline = capture_dataset_runtime_baseline(args)

        apply_dataset_runtime_policy(
            args,
            "Qwen2.5-VL-3B-Instruct-Replay",
            "LogicVista",
            runtime_baseline=baseline,
        )
        self.assertEqual(os.environ["QWEN2VL_VLLM_MAX_TOKENS"], "2048")

        apply_dataset_runtime_policy(
            args,
            "Qwen2.5-VL-3B-Instruct-Replay",
            "CountQA",
            runtime_baseline=baseline,
        )
        self.assertEqual(os.environ["VLLM_USE_V1"], "1")
        self.assertNotIn("QWEN2VL_VLLM_MAX_TOKENS", os.environ)

    def test_global_qwen_paths_do_not_reclassify_current_model(self):
        previous_path = os.environ.get("MODEL_PATH_QWEN25_3B")
        os.environ["MODEL_PATH_QWEN25_3B"] = "/models/Qwen2.5-VL-3B-Instruct"
        if previous_path is None:
            self.addCleanup(os.environ.pop, "MODEL_PATH_QWEN25_3B", None)
        else:
            self.addCleanup(os.environ.__setitem__, "MODEL_PATH_QWEN25_3B", previous_path)
        args = Namespace(batch_size=64)
        baseline = capture_dataset_runtime_baseline(args)

        self.assertFalse(_is_qwen25vl_model("Gemma3-12B-Replay"))
        policy = apply_dataset_runtime_policy(
            args,
            "Gemma3-12B-Replay",
            "LogicVista",
            runtime_baseline=baseline,
        )

        self.assertEqual(policy[0], "default")
        self.assertEqual(args.batch_size, 64)
        self.assertNotIn("QWEN2VL_VLLM_MAX_TOKENS", os.environ)

    def test_custom_config_alias_uses_current_entry_identity(self):
        config = {
            "class": "Qwen2VLChatReplay",
            "model_path": "/models/Qwen2.5-VL-3B-Instruct",
        }
        args = Namespace(batch_size=64)
        baseline = capture_dataset_runtime_baseline(args)

        self.assertTrue(_is_qwen25vl_model("checkpoint_A", model_config=config))
        policy = apply_dataset_runtime_policy(
            args,
            "checkpoint_A",
            "LogicVista",
            runtime_baseline=baseline,
            model_config=config,
        )

        self.assertEqual(policy[0], "qwen25vl_logicvista_v0")
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(os.environ["QWEN2VL_VLLM_MAX_TOKENS"], "2048")


if __name__ == "__main__":
    unittest.main()
