import importlib.util
import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd
from tempfile import TemporaryDirectory


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'validate_perception_full_results.py'


def _load_validator():
    spec = importlib.util.spec_from_file_location('perception_full_validator', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(
    path,
    expected_rows,
    matrix,
    task_identity,
    artifact_type,
    prediction_file=None,
    score_file=None,
):
    payload = {
        'schema_version': 2,
        'status': 'complete',
        'expected_rows': expected_rows,
        'matrix': matrix,
        'artifact_type': artifact_type,
        'task': task_identity,
        'model': {'key': task_identity['model_key']},
    }
    if prediction_file is not None:
        payload['prediction_file'] = str(prediction_file)
        payload['prediction_sha256'] = hashlib.sha256(
            Path(prediction_file).read_bytes()
        ).hexdigest()
    if score_file is not None:
        payload.update({
            'returncode': 0,
            'score_files': [str(score_file)],
            'score_file_sha256': {
                str(score_file): hashlib.sha256(Path(score_file).read_bytes()).hexdigest()
            },
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def _build_cell(root, mode, model, dataset, rows, matrix):
    task_root = root / 'default' / mode / 'baseline' / model / dataset
    task_identity = {
        'model_key': model,
        'policy_key': 'default',
        'mode': mode,
        'transform': 'baseline',
        'dataset': dataset,
    }
    prediction_file = task_root / 'predictions' / 'predictions.csv'
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'index': range(rows), 'prediction': ['1'] * rows}).to_csv(
        prediction_file, index=False
    )

    score_file = task_root / 'eval' / f'{dataset}_acc.csv'
    score_file.parent.mkdir(parents=True, exist_ok=True)
    if dataset == 'CountQA':
        pd.DataFrame([{
            'Overall': 0.5,
            'correct': rows // 2,
            'annotations': rows,
            'predictions': rows,
            'missing': 0,
            'invalid_format': 0,
        }]).to_csv(score_file, index=False)
    elif dataset == 'SpatialMQA':
        pd.DataFrame([{
            'Overall': 0.5,
            'correct': rows // 2,
            'total': rows,
            'predictions': rows,
            'missing': 0,
        }]).to_csv(score_file, index=False)
    else:
        score_file = task_root / 'eval' / f'{dataset}_score.csv'
        pd.DataFrame([{
            'Split': 'RefCOCOg_test',
            'Precision@1': 50.0,
            'Average IoU': 0.4,
            'Format Compliance': 100.0,
            'Samples': rows,
        }]).to_csv(score_file, index=False)

    _write_manifest(
        task_root / 'predictions' / 'manifest.json',
        rows,
        matrix,
        task_identity,
        'prediction',
        prediction_file=prediction_file,
    )
    _write_manifest(
        task_root / 'eval' / 'manifest.json',
        rows,
        matrix,
        task_identity,
        'eval',
        prediction_file=prediction_file,
        score_file=score_file,
    )
    return task_root


def _build_matrix(root, validator, model='model_a'):
    matrix = 'test_matrix'
    validator.EXPECTED_ROWS = {'CountQA': 4, 'SpatialMQA': 6, 'RefCOCO': 8}
    for mode in validator.MODE_TO_CONDITION:
        for dataset, rows in validator.EXPECTED_ROWS.items():
            _build_cell(root, mode, model, dataset, rows, matrix)

    inputs = root / '_inputs'
    inputs.mkdir(parents=True)
    allowlist = inputs / 'refcocog_test_indices.txt'
    raw = ''.join(f'RefCOCOg_test_{index}\n' for index in range(8)).encode('utf-8')
    allowlist.write_bytes(raw)
    allowlist_manifest = inputs / 'refcocog_test_indices.manifest.json'
    allowlist_manifest.write_text(json.dumps({
        'dataset': 'RefCOCO',
        'split_value': 'RefCOCOg_test',
        'selected_count': 8,
        'allowlist_path': str(allowlist),
        'allowlist_sha256': hashlib.sha256(raw).hexdigest(),
    }), encoding='utf-8')
    return model, matrix, allowlist_manifest


class PerceptionFullValidatorTest(unittest.TestCase):
    def test_validate_accepts_complete_matrix(self):
        validator = _load_validator()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model, matrix, allowlist_manifest = _build_matrix(root, validator)

            records, allowlist_payload = validator.validate(
                root, [model], matrix, allowlist_manifest
            )

        self.assertEqual(len(records), 6)
        self.assertEqual({row['condition'] for row in records}, {'IQ', 'IQIQ'})
        self.assertEqual(
            {row['dataset'] for row in records},
            {'CountQA', 'SpatialMQA', 'RefCOCOg_test'},
        )
        self.assertEqual(allowlist_payload['selected_count'], 8)

    def test_validate_rejects_wrong_prediction_denominator(self):
        validator = _load_validator()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model, matrix, allowlist_manifest = _build_matrix(root, validator)
            prediction_file = (
                root / 'default' / 'image_text' / 'baseline' / model / 'CountQA'
                / 'predictions' / 'predictions.csv'
            )
            pd.read_csv(prediction_file).iloc[:-1].to_csv(prediction_file, index=False)

            with self.assertRaisesRegex(AssertionError, 'Prediction denominator mismatch'):
                validator.validate(root, [model], matrix, allowlist_manifest)

    def test_validate_rejects_prediction_changed_after_eval(self):
        validator = _load_validator()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model, matrix, allowlist_manifest = _build_matrix(root, validator)
            prediction_file = (
                root / 'default' / 'image_text' / 'baseline' / model / 'CountQA'
                / 'predictions' / 'predictions.csv'
            )
            predictions = pd.read_csv(prediction_file)
            predictions['prediction'] = ['2'] * len(predictions)
            predictions.to_csv(prediction_file, index=False)

            with self.assertRaisesRegex(AssertionError, 'checksum mismatch'):
                validator.validate(root, [model], matrix, allowlist_manifest)


if __name__ == '__main__':
    unittest.main()
