import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vlmeval.probes.run_generation_attention import native_greedy_params
from vlmeval.probes.score_generation_attention import merge_records

BASE_ENV = {'VLMEVAL_LAZY_INIT':'1', 'VLMEVAL_VLM_MINIMAL_IMPORT':'1',
            'VLMEVAL_API_MINIMAL_IMPORT':'1'}

class SamplingAndCoverageTests(unittest.TestCase):
    def test_native_greedy_and_global_seed(self):
        args = SimpleNamespace(max_tokens=2048, limit=2, seed=10, offset=112, logprobs=0)
        with patch.dict(os.environ, BASE_ENV, clear=True):
            params, effective = native_greedy_params(args)
        self.assertEqual(effective['temperature'], 0)
        self.assertEqual(effective['repetition_penalty'], 1)
        self.assertEqual([p.seed for p in params], [122,123])
        self.assertTrue(all(p.logprobs is None and p.max_tokens == 2048 for p in params))

    def test_non_greedy_environment_rejected(self):
        args = SimpleNamespace(max_tokens=2048, limit=1, seed=10, offset=0, logprobs=0)
        with patch.dict(os.environ, dict(BASE_ENV, QWEN2VL_VLLM_TEMPERATURE='0.7'), clear=True):
            with self.assertRaises(ValueError):
                native_greedy_params(args)

    def test_merge_requires_exact_unique_original_ids(self):
        self.assertEqual(merge_records([[{'index':'2'}],[{'index':'1'}]], ['1','2']),
                         [{'index':'1'},{'index':'2'}])
        for groups in [[[{'index':'1'}]], [[{'index':'1'}],[{'index':'1'}]]]:
            with self.assertRaises(ValueError):
                merge_records(groups, ['1','2'])


if __name__ == '__main__':
    unittest.main()
