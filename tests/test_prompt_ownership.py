import os
import unittest


os.environ.setdefault("VLMEVAL_LAZY_INIT", "1")
os.environ.setdefault("VLMEVAL_VLM_MINIMAL_IMPORT", "1")

from vlmeval.inference import _maybe_build_prompt_struct, _replay_sample_indices


class _DatasetOwnedPrompt:
    FORCE_DATASET_PROMPT = True
    TYPE = "VQA"

    def build_prompt(self, row):
        return [{"type": "text", "value": f"dataset:{row['question']}"}]


class _ModelOwnedPrompt:
    def use_custom_prompt(self, dataset_name):
        return True

    def build_prompt(self, row, dataset=None):
        return [{"type": "text", "value": f"model:{row['question']}"}]


class PromptOwnershipTest(unittest.TestCase):
    def test_replay_index_context_restores_environment_and_propagates_errors(self):
        raw_key = "REPLAY_RAW_DUMP_PATH"
        index_key = "REPLAY_SAMPLE_INDICES_JSON"
        old_raw = os.environ.get(raw_key)
        old_indices = os.environ.get(index_key)
        os.environ[raw_key] = "/tmp/sentinel.jsonl"
        os.environ[index_key] = '["outer"]'
        try:
            with self.assertRaisesRegex(RuntimeError, "sentinel failure"):
                with _replay_sample_indices(["a", 2]):
                    self.assertEqual(os.environ[index_key], '["a", "2"]')
                    raise RuntimeError("sentinel failure")
            self.assertEqual(os.environ[index_key], '["outer"]')
            os.environ.pop(raw_key)
            with self.assertRaisesRegex(RuntimeError, "disabled failure"):
                with _replay_sample_indices(["ignored"]):
                    raise RuntimeError("disabled failure")
        finally:
            if old_raw is None:
                os.environ.pop(raw_key, None)
            else:
                os.environ[raw_key] = old_raw
            if old_indices is None:
                os.environ.pop(index_key, None)
            else:
                os.environ[index_key] = old_indices

    def test_dataset_protocol_wins_over_model_and_common_prompt(self):
        old = os.environ.get("REPLAY_FORCE_COMMON_PROMPT")
        os.environ["REPLAY_FORCE_COMMON_PROMPT"] = "1"
        try:
            prompt = _maybe_build_prompt_struct(
                _ModelOwnedPrompt(),
                _DatasetOwnedPrompt(),
                "ProtocolCriticalDataset",
                {"question": "sentinel"},
            )
        finally:
            if old is None:
                os.environ.pop("REPLAY_FORCE_COMMON_PROMPT", None)
            else:
                os.environ["REPLAY_FORCE_COMMON_PROMPT"] = old

        self.assertEqual(prompt, [{"type": "text", "value": "dataset:sentinel"}])


if __name__ == "__main__":
    unittest.main()
