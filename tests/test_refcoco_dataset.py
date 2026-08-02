import os

import numpy as np
import pandas as pd
from PIL import Image


def _dataset(tmp_path, monkeypatch):
    from vlmeval.dataset.refcoco import RefCOCODataset

    image = tmp_path / 'refcoco.jpg'
    Image.new('RGB', (640, 376)).save(image)
    dataset = object.__new__(RefCOCODataset)
    dataset.dataset_name = 'RefCOCO'
    dataset.meta_only = True
    dataset.data = pd.DataFrame([{
        'index': 'RefCOCOg_test_42959',
        'image_path': os.fspath(image),
        'question': (
            'Please provide the bounding box coordinate of the region this sentence describes: '
            '<ref>the man in yellow coat</ref>'
        ),
        'width': 1.0,
        'height': 1.0,
        'bbox_x1': 0.584859,
        'bbox_y1': 0.173032,
        'bbox_x2': 0.797422,
        'bbox_y2': 0.710106,
        'split': 'RefCOCOg_test',
    }])
    monkeypatch.setenv('REFCOCO_COORDINATE_MODE', 'normalized_0_1_xyxy')
    return dataset


def test_refcocog_normalized_prompt_matches_executable_protocol(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path, monkeypatch)
    prompt = dataset.build_prompt(0)

    assert [item['type'] for item in prompt] == ['image', 'text']
    text = prompt[-1]['value']
    assert '(top-left x, top-left y, bottom-right x, bottom-right y)' in text
    assert 'bounded between 0 and 1' in text
    assert 'exact format [x1, y1, x2, y2]' in text
    assert text.endswith('the man in yellow coat')
    assert '<ref>' not in text


def test_refcocog_normalized_mode_rejects_other_coordinate_systems(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path, monkeypatch)
    assert dataset._parse_model_prediction('bbox: 0.1 0.2 0.8 0.9') is None
    assert dataset._parse_model_prediction('bbox: [0.1, 0.2, 0.8, 0.9]') is None
    assert dataset._parse_model_prediction('[0.1, 0.2, 0.8, 0.9] extra') is None
    assert dataset._parse_model_prediction('[0.1, 0.2, 0.8, 0.9] [0.1, 0.2, 0.8, 0.9]') is None
    assert np.allclose(
        dataset._parse_model_prediction('[0.1, 0.2, 0.8, 0.9]'),
        [0.1, 0.2, 0.8, 0.9],
    )
    accepted = dataset._prediction_to_absolute(
        np.array([0.584859, 0.173032, 0.797422, 0.710106]),
        640,
        376,
    )
    assert np.allclose(accepted, [374.30976, 65.059, 510.35008, 267.0], atol=1e-3)
    assert dataset._prediction_to_absolute(np.array([584, 173, 797, 710]), 640, 376) is None
    assert dataset._prediction_to_absolute(np.array([0.8, 0.2, 0.5, 0.7]), 640, 376) is None


def test_refcocog_gemma_native_protocol_prompt_parse_and_axis_conversion(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path, monkeypatch)
    monkeypatch.setenv('REFCOCO_COORDINATE_MODE', 'gemma_0_1000_yxyx')

    prompt = dataset.build_prompt(0)[-1]['value']
    assert prompt == 'detect the man in yellow coat, output only ```json'
    assert 'the man in yellow coat' in prompt
    assert '<ref>' not in prompt

    parsed = dataset._parse_model_prediction(
        '```json\n[{"box_2d": [173, 585, 710, 797], "label": "man"}]\n```'
    )
    assert np.allclose(parsed, [173, 585, 710, 797])
    absolute = dataset._prediction_to_absolute(parsed, 640, 376)
    assert np.allclose(absolute, [374.4, 65.048, 510.08, 266.96], atol=1e-3)

    assert dataset._parse_model_prediction('[{"box_2d": [173, 585, 710, 797]}]') is None
    assert dataset._parse_model_prediction('```json\n[{"box_2d": [173, 585, 710]}]\n```') is None
    assert dataset._parse_model_prediction('```json\n[{"box_2d": [173.0, 585, 710, 797]}]\n```') is None
    assert dataset._parse_model_prediction('```json\n[{"box_2d": [true, 585, 710, 797]}]\n```') is None
    assert dataset._parse_model_prediction('```json\n[{"box_2d": ["173", 585, 710, 797]}]\n```') is None
    assert dataset._parse_model_prediction(
        '```json\n[{"box_2d": [173, 585, 710, 797]}, {"box_2d": [1, 2, 3, 4]}]\n```'
    ) is None
    assert dataset._prediction_to_absolute(np.array([173, 585, 710, 1001]), 640, 376) is None
    assert dataset._prediction_to_absolute(np.array([710, 585, 173, 797]), 640, 376) is None


def test_refcocog_minicpm_v46_protocol_prompt_parse_and_conversion(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path, monkeypatch)
    monkeypatch.setenv('REFCOCO_COORDINATE_MODE', 'minicpm_v46_0_1000_xyxy')

    prompt = dataset.build_prompt(0)[-1]['value']
    assert prompt == (
        'Please provide the bounding box coordinate of the region this sentence describes: '
        '<ref>the man in yellow coat</ref>'
    )

    parsed = dataset._parse_model_prediction('<box>585 173 797 710</box>')
    assert np.allclose(parsed, [585, 173, 797, 710])
    absolute = dataset._prediction_to_absolute(parsed, 640, 376)
    assert np.allclose(absolute, [374.4, 65.048, 510.08, 266.96], atol=1e-3)

    assert dataset._parse_model_prediction('<box>585, 173, 797, 710</box>') is None
    assert np.allclose(
        dataset._parse_model_prediction('box: <box>585 173 797 710</box>'),
        [585, 173, 797, 710],
    )
    assert np.allclose(dataset._parse_model_prediction(
        '<box>585 173 797 710</box><box>1 2 3 4</box>'
    ), [585, 173, 797, 710])
    assert dataset._prediction_to_absolute(np.array([585, 173, 1001, 710]), 640, 376) is None
    assert dataset._prediction_to_absolute(np.array([797, 173, 585, 710]), 640, 376) is None


def test_refcocog_legacy_coordinate_mode_remains_default(monkeypatch):
    from vlmeval.dataset.refcoco import RefCOCODataset

    monkeypatch.delenv('REFCOCO_COORDINATE_MODE', raising=False)
    dataset = object.__new__(RefCOCODataset)
    assert dataset.coordinate_mode() == 'legacy_auto'
    assert np.allclose(
        dataset._prediction_to_absolute(np.array([500, 250, 750, 750]), 640, 376),
        [320, 94, 480, 282],
    )


def test_refcocog_strict_evaluate_scores_only_valid_normalized_boxes(tmp_path, monkeypatch):
    from vlmeval.dataset.refcoco import RefCOCODataset
    from vlmeval.smp.file import get_intermediate_file_path, load

    monkeypatch.setenv('REFCOCO_COORDINATE_MODE', 'normalized_0_1_xyxy')
    dataset = object.__new__(RefCOCODataset)
    dataset.dataset_name = 'RefCOCO'
    dataset._ensure_metadata_ready = lambda: None
    dataset.data = pd.DataFrame([
        {
            'index': 'valid',
            'width': 1.0,
            'height': 1.0,
            'bbox_x1': 0.1,
            'bbox_y1': 0.2,
            'bbox_x2': 0.8,
            'bbox_y2': 0.9,
            'split': 'RefCOCOg_test',
        },
        {
            'index': 'invalid',
            'width': 1.0,
            'height': 1.0,
            'bbox_x1': 0.1,
            'bbox_y1': 0.2,
            'bbox_x2': 0.8,
            'bbox_y2': 0.9,
            'split': 'RefCOCOg_test',
        },
    ])
    prediction_file = tmp_path / 'refcoco.csv'
    pd.DataFrame([
        {'index': 'valid', 'prediction': '[0.1, 0.2, 0.8, 0.9]'},
        {'index': 'invalid', 'prediction': '[100, 200, 800, 900]'},
    ]).to_csv(prediction_file, index=False)

    summary = dataset.evaluate(prediction_file)
    row = summary[summary['Split'] == 'RefCOCOg_test'].iloc[0]
    assert row['Precision@1'] == 50.0
    assert row['Average IoU'] == 0.5
    assert row['Format Compliance'] == 50.0
    assert row['Samples'] == 2

    detail_file = get_intermediate_file_path(os.fspath(prediction_file), '_detail')
    detail = pd.DataFrame(load(detail_file))
    assert detail['pred_bbox_valid'].tolist() == [1, 0]
    assert set(detail['coordinate_mode']) == {'normalized_0_1_xyxy'}
