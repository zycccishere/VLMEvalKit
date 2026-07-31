import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('VLMEVAL_LAZY_INIT', '1')
os.environ.setdefault('VLMEVAL_VLM_MINIMAL_IMPORT', '1')
os.environ.setdefault('VLMEVAL_API_MINIMAL_IMPORT', '1')
import vlmeval  # noqa: E402,F401

dataset_package = types.ModuleType('vlmeval.dataset')
dataset_package.__path__ = [os.fspath(REPO_ROOT / 'vlmeval' / 'dataset')]
sys.modules.setdefault('vlmeval.dataset', dataset_package)

import vlmeval.dataset.countqa as countqa_module  # noqa: E402
from vlmeval.dataset.countqa import (  # noqa: E402
    CountQADataset,
    expand_countqa_rows,
    make_countqa_index,
    parse_countqa_integer,
    score_countqa_predictions,
)


class CountQADatasetTest(unittest.TestCase):
    def test_expands_image_rows_and_preserves_metadata(self):
        rows = [{
            'questions': ['How many empty trays?', 'How many tiles?'],
            'answers': ['0', '400'],
            'objects': ['tray', 'tile'],
            'categories': ['synthetic'],
            'is_focused': False,
            'image_path': 'countqa-0000.jpg',
        }]

        expanded = expand_countqa_rows(rows)

        self.assertEqual(len(expanded), 2)
        self.assertEqual(
            [row['index'] for row in expanded],
            [make_countqa_index(0, 0), make_countqa_index(0, 1)],
        )
        self.assertEqual([row['answer'] for row in expanded], ['0', '400'])
        self.assertEqual(expanded[1]['objects'], ['tray', 'tile'])
        self.assertEqual(expanded[1]['categories'], ['synthetic'])
        self.assertFalse(expanded[1]['is_focused'])

    def test_rejects_unequal_question_answer_lists(self):
        rows = [{
            'questions': ['q0', 'q1'],
            'answers': ['0'],
            'objects': [],
            'categories': [],
            'is_focused': True,
        }]
        with self.assertRaisesRegex(ValueError, '2 questions but 1 answers'):
            expand_countqa_rows(rows)

    def test_strict_integer_parser_accepts_boundaries_and_rejects_ambiguity(self):
        self.assertEqual(parse_countqa_integer('0'), 0)
        self.assertEqual(parse_countqa_integer('400'), 400)
        self.assertEqual(parse_countqa_integer(' 400\n'), 400)
        for prediction in ('There are 2.', '2 or 3', '2.0', 2.0, '-1', '[2]', '', 'two'):
            with self.subTest(prediction=prediction):
                self.assertIsNone(parse_countqa_integer(prediction))

    def test_scores_by_canonical_index_and_marks_missing_prediction(self):
        annotations = pd.DataFrame([
            {'index': make_countqa_index(0, 0), 'answer': '0'},
            {'index': make_countqa_index(1, 0), 'answer': '400'},
        ])
        predictions = pd.DataFrame([
            {'index': make_countqa_index(1, 0), 'prediction': '400'},
        ])

        details, summary = score_countqa_predictions(annotations, predictions)

        self.assertEqual(details['exact_match'].tolist(), [0, 1])
        self.assertEqual(details['parse_status'].tolist(), ['missing', 'valid'])
        self.assertEqual(summary.iloc[0]['score'], 0.5)
        self.assertEqual(summary.iloc[0]['accuracy_percent'], 50.0)
        self.assertEqual(summary.iloc[0]['missing'], 1)

    def test_rejects_duplicate_prediction_index(self):
        index = make_countqa_index(0, 0)
        annotations = pd.DataFrame([{'index': index, 'answer': '0'}])
        predictions = pd.DataFrame([
            {'index': index, 'prediction': '0'},
            {'index': index, 'prediction': '400'},
        ])
        with self.assertRaisesRegex(ValueError, 'duplicate canonical index'):
            score_countqa_predictions(annotations, predictions)

    def test_evaluate_preserves_decimal_text_across_csv_and_xlsx(self):
        index = make_countqa_index(0, 0)
        dataset = object.__new__(CountQADataset)
        dataset.data = pd.DataFrame([{'index': index, 'answer': '2'}])
        predictions = pd.DataFrame([{'index': index, 'prediction': '2.0'}])

        with tempfile.TemporaryDirectory() as temporary_directory:
            for suffix in ('csv', 'xlsx'):
                path = Path(temporary_directory) / f'predictions.{suffix}'
                if suffix == 'csv':
                    predictions.to_csv(path, index=False)
                else:
                    predictions.to_excel(path, index=False)
                score = dataset.evaluate(path)
                with self.subTest(suffix=suffix):
                    self.assertEqual(score.iloc[0]['accuracy_percent'], 0.0)
                    self.assertEqual(score.iloc[0]['invalid_format'], 1)

    def test_build_prompt_is_exactly_image_then_integer_only_text(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / 'countqa-0000.jpg'
            image_path.write_bytes(b'synthetic-image-placeholder')
            dataset = object.__new__(CountQADataset)
            dataset.img_root = temporary_directory

            prompt = dataset.build_prompt({
                'index': make_countqa_index(0, 0),
                'image_path': image_path.name,
                'question': 'How many objects are visible?',
            })

        self.assertEqual([message['type'] for message in prompt], ['image', 'text'])
        self.assertEqual(len(prompt), 2)
        self.assertEqual(prompt[0]['value'], os.fspath(image_path))
        self.assertIn('exactly one non-negative integer', prompt[1]['value'])
        self.assertIn('integer only', prompt[1]['value'])

    def test_load_requires_revision_and_checksum_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = object.__new__(CountQADataset)
            with patch.object(countqa_module, 'LMUDataRoot', return_value=temporary_directory):
                with self.assertRaisesRegex(FileNotFoundError, 'manifest'):
                    dataset.load_data('CountQA')

    def test_load_verifies_table_and_image_inventory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / 'images' / 'CountQA'
            image_dir.mkdir(parents=True)
            image_path = image_dir / 'countqa-0000.jpg'
            image_path.write_bytes(b'image-bytes')
            table_path = root / 'CountQA.tsv'
            pd.DataFrame([{
                'index': make_countqa_index(0, 0),
                'source_image_row': 0,
                'question_ordinal': 0,
                'question': 'How many?',
                'answer': '2',
                'image_path': image_path.name,
                'objects': '[]',
                'categories': '[]',
                'is_focused': True,
            }]).to_csv(table_path, sep='\t', index=False)
            manifest = {
                'dataset_id': countqa_module.COUNTQA_DATASET_ID,
                'revision': countqa_module.COUNTQA_REVISION,
                'counts': {'image_rows': 1, 'qa_rows': 1},
                'tables': [{
                    'path': table_path.name,
                    'sha256': countqa_module._sha256_file(table_path),
                }],
                'images': [{
                    'path': os.fspath(image_path.relative_to(root)),
                    'bytes': image_path.stat().st_size,
                    'sha256': countqa_module._sha256_file(image_path),
                }],
            }
            (root / 'CountQA.manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            dataset = object.__new__(CountQADataset)
            with (
                patch.object(countqa_module, 'LMUDataRoot', return_value=temporary_directory),
                patch.object(countqa_module, 'COUNTQA_IMAGE_ROWS', 1),
                patch.object(countqa_module, 'COUNTQA_QA_ROWS', 1),
            ):
                loaded = dataset.load_data('CountQA')
                self.assertEqual(len(loaded), 1)
                image_path.write_bytes(b'tampered-image-bytes')
                with self.assertRaisesRegex(ValueError, 'size mismatch'):
                    dataset.load_data('CountQA')

                image_path.write_bytes(b'image-bytes')
                escaped_path = root.parent / f'{root.name}-escaped.jpg'
                escaped_path.write_bytes(b'image-bytes')
                manifest['images'][0] = {
                    'path': f'../{escaped_path.name}',
                    'bytes': escaped_path.stat().st_size,
                    'sha256': countqa_module._sha256_file(escaped_path),
                }
                (root / 'CountQA.manifest.json').write_text(
                    json.dumps(manifest), encoding='utf-8'
                )
                with self.assertRaisesRegex(ValueError, 'escapes LMUData'):
                    dataset.load_data('CountQA')


if __name__ == '__main__':
    unittest.main()
