import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vlmeval.cli.run_benchmark import BenchmarkRunner


REPO_ROOT = Path(__file__).resolve().parents[1]


class DatasetTaskEnvTest(unittest.TestCase):
    def _runner(self, lmu_data):
        args = argparse.Namespace(
            matrix_config=REPO_ROOT / 'configs' / 'matrix_perception_benchmarks_smoke_8gpu_20260731.yaml',
            model_config=REPO_ROOT / 'configs' / 'models.yaml',
            nodes=1,
            node_rank=0,
            gpu_ids='0,1,2,3,4,5,6,7',
            models='',
            policies='',
            modes='',
            transforms='',
            datasets='',
            task_manifest=None,
            scheduler='gpu_pool',
            manifest_is_node_shard=False,
            resume_infer=None,
            plan_only=True,
        )
        env = {
            'LMUData': lmu_data,
            'MODEL_ROOT': '/models',
            'CONDA_ROOT': '/conda',
        }
        with patch.dict(os.environ, env, clear=False):
            return BenchmarkRunner(REPO_ROOT, args)

    @staticmethod
    def _task(runner, model_key, dataset):
        return next(
            task for task in runner.tasks
            if task.model_key == model_key
            and task.dataset == dataset
            and task.mode == 'image_text'
        )

    def test_countqa_generation_override_is_dataset_scoped(self):
        with tempfile.TemporaryDirectory() as lmu_data:
            runner = self._runner(lmu_data)

        qwen = runner.models['qwen25vl_3b']
        qwen_count = runner.build_env(qwen, self._task(runner, qwen.key, 'CountQA'), ['0'])
        qwen_spatial = runner.build_env(qwen, self._task(runner, qwen.key, 'SpatialMQA'), ['0'])
        self.assertEqual(qwen_count['QWEN2VL_VLLM_TEMPERATURE'], '0')
        self.assertEqual(qwen_count['QWEN2VL_VLLM_TOP_P'], '1')
        self.assertEqual(qwen_count['QWEN2VL_VLLM_MAX_TOKENS'], '32')
        self.assertNotIn('QWEN2VL_VLLM_MAX_TOKENS', qwen_spatial)

        gemma = runner.models['gemma3_4b']
        gemma_count = runner.build_env(gemma, self._task(runner, gemma.key, 'CountQA'), ['1'])
        gemma_spatial = runner.build_env(gemma, self._task(runner, gemma.key, 'SpatialMQA'), ['1'])
        self.assertEqual(gemma_count['GEMMA3_MAX_NEW_TOKENS'], '32')
        self.assertEqual(gemma_count['GEMMA3_VLLM_TEMPERATURE'], '0')
        self.assertEqual(gemma_spatial['GEMMA3_MAX_NEW_TOKENS'], '4096')

        minicpm = runner.models['minicpm_o_45_no_reasoning']
        minicpm_count = runner.build_env(minicpm, self._task(runner, minicpm.key, 'CountQA'), ['2'])
        minicpm_spatial = runner.build_env(minicpm, self._task(runner, minicpm.key, 'SpatialMQA'), ['2'])
        self.assertEqual(minicpm_count['MINICPM45_MAX_NEW_TOKENS'], '32')
        self.assertEqual(minicpm_count['MINICPM45_NUM_BEAMS'], '1')
        self.assertEqual(minicpm_count['MINICPM45_SAMPLING'], '0')
        self.assertEqual(minicpm_count['MINICPM45_REPETITION_PENALTY'], '1')
        self.assertEqual(minicpm_spatial['MINICPM45_MAX_NEW_TOKENS'], '16384')
        for key in (
            'MINICPM45_NUM_BEAMS',
            'MINICPM45_SAMPLING',
            'MINICPM45_TEMPERATURE',
            'MINICPM45_TOP_P',
            'MINICPM45_TOP_K',
            'MINICPM45_REPETITION_PENALTY',
            'MINICPM45_PRESENCE_PENALTY',
        ):
            self.assertNotIn(key, minicpm_spatial)

        qwen_refcoco = runner.build_env(qwen, self._task(runner, qwen.key, 'RefCOCO'), ['3'])
        self.assertEqual(qwen_refcoco['REFCOCO_COORDINATE_MODE'], 'normalized_0_1_xyxy')
        self.assertNotIn('REFCOCO_COORDINATE_MODE', qwen_spatial)


if __name__ == '__main__':
    unittest.main()
