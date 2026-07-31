import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image


os.environ.setdefault('VLMEVAL_LAZY_INIT', '1')
os.environ.setdefault('VLMEVAL_VLM_MINIMAL_IMPORT', '1')
os.environ.setdefault('VLMEVAL_API_MINIMAL_IMPORT', '1')

import vlmeval  # noqa: E402,F401


# Load the assigned dataset module without importing every unrelated benchmark.
dataset_package = types.ModuleType('vlmeval.dataset')
dataset_package.__path__ = [str(Path(__file__).resolve().parents[1] / 'vlmeval' / 'dataset')]
dataset_package.__package__ = 'vlmeval.dataset'
sys.modules['vlmeval.dataset'] = dataset_package

from vlmeval.dataset.ref_l4 import (  # noqa: E402
    BBoxProtocolError,
    ParsedBBox,
    RefL4Dataset,
    bbox_iou,
    convert_bbox_to_original_pixels,
    parse_bbox_protocol,
    parse_prediction_to_original_pixels,
    xywh_to_xyxy,
)


def protocol(
    bbox,
    *,
    coordinate_space='normalized_1000',
    coordinate_size=(1000, 1000),
    bbox_format='xyxy',
):
    return json.dumps({
        'bbox_2d': bbox,
        'bbox_format': bbox_format,
        'coordinate_space': coordinate_space,
        'coordinate_size': list(coordinate_size),
    })


def canonical_row(annotation_id, bbox=(0, 0, 10, 10), size=(100, 100)):
    return {
        'index': str(annotation_id),
        'annotation_id': annotation_id,
        'question': 'the target region',
        'image_path': '/unused/by/evaluate.jpg',
        'bbox_x': bbox[0],
        'bbox_y': bbox[1],
        'bbox_w': bbox[2],
        'bbox_h': bbox[3],
        'bbox_area': bbox[2] * bbox[3],
        'bbox_id': f'bbox-{annotation_id}',
        'ori_category_id': 'category-1',
        'image_id': 'image-1',
        'height': size[1],
        'width': size[0],
        'is_rewrite': False,
        'split': 'val',
        'source_revision': 'sentinel',
    }


def dataset_from_rows(rows):
    dataset = RefL4Dataset.__new__(RefL4Dataset)
    dataset.dataset_name = 'RefL4'
    dataset.data = pd.DataFrame(rows)
    return dataset


class RefL4CoordinateProtocolTest(unittest.TestCase):
    def test_xywh_to_xyxy_and_coordinate_order(self):
        self.assertEqual(xywh_to_xyxy([10, 20, 30, 40]), (10.0, 20.0, 40.0, 60.0))
        with self.assertRaisesRegex(BBoxProtocolError, 'positive width'):
            xywh_to_xyxy([10, 20, 0, 40])
        with self.assertRaisesRegex(BBoxProtocolError, 'non-degenerate'):
            parse_bbox_protocol(protocol([20, 10, 10, 30]))

    def test_normalized_coordinates_scale_to_canonical_size(self):
        parsed = parse_bbox_protocol(protocol([100, 200, 300, 400]))
        self.assertEqual(
            convert_bbox_to_original_pixels(parsed, original_size=(100, 50)),
            (10.0, 10.0, 30.0, 20.0),
        )
        self.assertEqual(
            convert_bbox_to_original_pixels(parsed, original_size=(200, 50)),
            (20.0, 10.0, 60.0, 20.0),
        )

    def test_only_fixed_normalized_canvas_is_accepted(self):
        parsed = ParsedBBox(
            bbox_xyxy=(100.0, 250.0, 800.0, 750.0),
            coordinate_space='normalized_1000',
            coordinate_size=(1000.0, 1000.0),
        )
        self.assertEqual(
            convert_bbox_to_original_pixels(parsed, original_size=(100, 60)),
            (10.0, 15.0, 80.0, 45.0),
        )
        with self.assertRaisesRegex(BBoxProtocolError, 'normalized_1000'):
            parse_bbox_protocol(protocol([10, 20, 30, 40], coordinate_space='original_pixel'))
        with self.assertRaisesRegex(BBoxProtocolError, r'exactly \[1000, 1000\]'):
            parse_bbox_protocol(protocol([10, 20, 30, 40], coordinate_size=(200, 120)))

    def test_parser_rejects_unproven_or_malformed_coordinates(self):
        bad_predictions = [
            '[10, 20, 30, 40]',
            '{"bbox_2d":[10,20,30,40]}',
            protocol([10, 20, 30]),
            protocol([10, 20, float('nan'), 40]),
            protocol([10, 20, 30, 40], bbox_format='xywh'),
            'answer: ' + protocol([10, 20, 30, 40]),
        ]
        for prediction in bad_predictions:
            with self.subTest(prediction=prediction), self.assertRaises(BBoxProtocolError):
                parse_prediction_to_original_pixels(prediction, original_size=(100, 50))

    def test_parser_rejects_additional_json_fields(self):
        payload = json.loads(protocol([10, 20, 30, 40]))
        payload['explanation'] = 'extra'
        with self.assertRaisesRegex(BBoxProtocolError, 'keys must be exactly'):
            parse_bbox_protocol(json.dumps(payload))

    def test_parser_rejects_duplicate_keys_and_preserves_out_of_bounds_domain(self):
        duplicate = (
            '{"bbox_2d":[10,20,30,40],"bbox_2d":[20,30,40,50],'
            '"bbox_format":"xyxy","coordinate_space":"normalized_1000",'
            '"coordinate_size":[1000,1000]}'
        )
        with self.assertRaisesRegex(BBoxProtocolError, 'duplicate key'):
            parse_bbox_protocol(duplicate)
        self.assertEqual(
            parse_prediction_to_original_pixels(
                protocol([-10, -20, 1010, 1020]),
                original_size=(100, 50),
            ),
            (-1.0, -1.0, 101.0, 51.0),
        )


class RefL4MetricTest(unittest.TestCase):
    def test_iou_uses_continuous_area_and_threshold_is_strict(self):
        self.assertEqual(bbox_iou((0, 0, 10, 10), (0, 0, 5, 10)), 0.5)
        dataset = dataset_from_rows([canonical_row(7)])
        with tempfile.TemporaryDirectory() as directory:
            eval_file = os.path.join(directory, 'predictions.csv')
            pd.DataFrame([
                {'index': '7', 'prediction': protocol([0, 0, 50, 100])},
            ]).to_csv(eval_file, index=False)
            scores = dataset.evaluate(eval_file)
        self.assertEqual(scores.loc[0, 'Ann-level acc iou 0.5'], 0.0)
        self.assertEqual(scores.loc[0, 'Ann-level macc iou 0.5:0.95'], 0.0)

    def test_evaluation_joins_canonical_gold_and_ignores_forged_gold_columns(self):
        dataset = dataset_from_rows([canonical_row(11, bbox=(10, 20, 30, 40), size=(100, 80))])
        with tempfile.TemporaryDirectory() as directory:
            eval_file = os.path.join(directory, 'predictions.csv')
            pd.DataFrame([{
                'index': '11',
                'prediction': protocol([100, 250, 400, 750]),
                'annotation_id': 999,
                'bbox_x': 999,
                'bbox_y': 999,
                'bbox_w': 1,
                'bbox_h': 1,
                'width': 1,
                'height': 1,
            }]).to_csv(eval_file, index=False)
            scores = dataset.evaluate(eval_file)
            details = pd.read_csv(os.path.join(directory, 'predictions_detail.csv'))

        self.assertEqual(scores.loc[0, 'Ann-level acc iou 0.5'], 100.0)
        self.assertEqual(json.loads(details.loc[0, 'gt_bbox_xyxy']), [10.0, 20.0, 40.0, 60.0])

    def test_evaluation_uses_normalized_canvas_independent_of_processor(self):
        dataset = dataset_from_rows([canonical_row(5, bbox=(20, 15, 60, 30), size=(100, 60))])
        normalized_prediction = protocol([200, 250, 800, 750])
        with tempfile.TemporaryDirectory() as directory:
            eval_file = os.path.join(directory, 'normalized.csv')
            pd.DataFrame([{
                'index': '5',
                'prediction': normalized_prediction,
            }]).to_csv(eval_file, index=False)
            scores = dataset.evaluate(eval_file)
        self.assertEqual(scores.loc[0, 'Ann-level acc iou 0.9'], 100.0)

    def test_evaluation_hard_fails_duplicate_and_extra_ids(self):
        two_row_dataset = dataset_from_rows([canonical_row(1), canonical_row(2)])
        cases = [
            (
                [
                    {'index': '1', 'prediction': protocol([0, 0, 100, 100])},
                    {'index': '1', 'prediction': protocol([0, 0, 100, 100])},
                ],
                'Duplicate prediction',
            ),
            (
                [
                    {'index': '1', 'prediction': protocol([0, 0, 100, 100])},
                    {'index': '2', 'prediction': protocol([0, 0, 100, 100])},
                    {'index': '3', 'prediction': protocol([0, 0, 100, 100])},
                ],
                'Unknown prediction',
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for offset, (rows, message) in enumerate(cases):
                eval_file = os.path.join(directory, f'bad-{offset}.csv')
                pd.DataFrame(rows).to_csv(eval_file, index=False)
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    two_row_dataset.evaluate(eval_file)

    def test_missing_and_invalid_predictions_remain_in_denominator(self):
        two_row_dataset = dataset_from_rows([canonical_row(1), canonical_row(2)])
        with tempfile.TemporaryDirectory() as directory:
            missing_file = os.path.join(directory, 'missing.csv')
            pd.DataFrame([
                {'index': '1', 'prediction': protocol([0, 0, 100, 100])},
            ]).to_csv(missing_file, index=False)
            missing_scores = two_row_dataset.evaluate(missing_file)
            missing_details = pd.read_csv(os.path.join(directory, 'missing_detail.csv'))

            self.assertEqual(missing_scores.loc[0, 'valid_parse_count'], 1)
            self.assertEqual(missing_scores.loc[0, 'valid_parse_rate'], 0.5)
            self.assertEqual(missing_scores.loc[0, 'missing_count'], 1)
            self.assertEqual(missing_scores.loc[0, 'invalid_count'], 0)
            self.assertEqual(missing_scores.loc[0, 'Ann-level acc iou 0.5'], 50.0)
            missing_row = missing_details[missing_details['index'] == 2].iloc[0]
            self.assertEqual(missing_row['missing_prediction'], 1)
            self.assertEqual(missing_row['iou'], 0.0)

            unproven_file = os.path.join(directory, 'unproven.csv')
            pd.DataFrame([
                {'index': '1', 'prediction': '[0, 0, 10, 10]'},
                {'index': '2', 'prediction': protocol([0, 0, 100, 100])},
            ]).to_csv(unproven_file, index=False)
            invalid_scores = two_row_dataset.evaluate(unproven_file)
            invalid_details = pd.read_csv(os.path.join(directory, 'unproven_detail.csv'))

            self.assertEqual(invalid_scores.loc[0, 'valid_parse_count'], 1)
            self.assertEqual(invalid_scores.loc[0, 'valid_parse_rate'], 0.5)
            self.assertEqual(invalid_scores.loc[0, 'missing_count'], 0)
            self.assertEqual(invalid_scores.loc[0, 'invalid_count'], 1)
            self.assertEqual(invalid_scores.loc[0, 'Ann-level acc iou 0.5'], 50.0)
            invalid_row = invalid_details[invalid_details['index'] == 1].iloc[0]
            self.assertEqual(invalid_row['parse_valid'], 0)
            self.assertEqual(invalid_row['iou'], 0.0)
        self.assertIn('exactly one JSON object', invalid_row['invalid_reason'])


class RefL4PromptTest(unittest.TestCase):
    def test_force_dataset_prompt_is_single_image_then_text(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, 'sentinel.png')
            Image.new('RGB', (100, 50), color='white').save(image_path)
            row = canonical_row(1, size=(100, 50))
            row['image_path'] = image_path
            dataset = dataset_from_rows([row])
            dataset.meta_only = True
            dataset.img_root = directory

            prompt = dataset.build_prompt(0)

        self.assertTrue(RefL4Dataset.FORCE_DATASET_PROMPT)
        self.assertEqual([item['type'] for item in prompt], ['image', 'text'])
        self.assertEqual(len(prompt), 2)
        self.assertIn('"coordinate_space":"normalized_1000"', prompt[1]['value'])
        self.assertIn('"coordinate_size":[1000,1000]', prompt[1]['value'])
        self.assertIn('independent of image resolution or internal resizing', prompt[1]['value'])


if __name__ == '__main__':
    unittest.main()
