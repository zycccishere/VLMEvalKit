import importlib
import json
import os
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

os.environ.setdefault('VLMEVAL_LAZY_INIT', '1')
import vlmeval  # noqa: E402


dataset_package = types.ModuleType('vlmeval.dataset')
dataset_package.__path__ = [str(Path(vlmeval.__file__).resolve().parent / 'dataset')]
sys.modules.setdefault('vlmeval.dataset', dataset_package)
SpatialMQA = importlib.import_module('vlmeval.dataset.spatialmqa').SpatialMQA


def write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (4, 3), color=(120, 80, 40)).save(path, format='JPEG')


def make_records():
    return [
        {
            'image': '000000000001.jpg',
            'question': 'Where is the object?',
            'options': ['left of', 'right of'],
            'answer': 'right of',
        },
        {
            'image': '000000000002.jpg',
            'question': 'Which spatial relation applies?',
            'options': ['on/above', 'below', 'in front of', 'behind', 'left of', 'right of'],
            'answer': 'F',
        },
    ]


def make_dataset(tmp_path):
    image_root = tmp_path / 'images'
    for record in make_records():
        write_image(image_root / record['image'])
    data = SpatialMQA._records_to_dataframe(make_records(), str(image_root))
    dataset = SpatialMQA.__new__(SpatialMQA)
    dataset.data = data
    dataset.dataset_name = 'SpatialMQA'
    dataset.meta_only = True
    return dataset


def test_canonicalizes_text_and_letter_gold_and_builds_single_turn_prompts(tmp_path):
    dataset = make_dataset(tmp_path)

    assert SpatialMQA.FORCE_DATASET_PROMPT is True
    assert dataset.data['answer'].tolist() == ['B', 'F']
    assert dataset.data['relation'].tolist() == ['right of', 'right of']
    assert dataset.data['category'].tolist() == ['x', 'x']
    assert dataset.data['index'].tolist() == [
        'spatialmqa-test-000000',
        'spatialmqa-test-000001',
    ]

    two_choice_prompt = dataset.build_prompt(0)
    six_choice_prompt = dataset.build_prompt(1)
    assert [message['type'] for message in two_choice_prompt] == ['image', 'text']
    assert [message['type'] for message in six_choice_prompt] == ['image', 'text']
    assert 'A. left of' in two_choice_prompt[1]['value']
    assert 'B. right of' in two_choice_prompt[1]['value']
    assert 'F. right of' in six_choice_prompt[1]['value']
    assert six_choice_prompt[1]['value'].endswith(
        'Respond with only the letter of the correct choice.'
    )


def test_prediction_parser_accepts_clear_letter_or_unique_text_and_rejects_conflict():
    options = ['left of', 'right of']
    assert SpatialMQA._parse_prediction('B', options) == 'B'
    assert SpatialMQA._parse_prediction('The answer is B.', options) == 'B'
    assert SpatialMQA._parse_prediction('right of', options) == 'B'
    assert SpatialMQA._parse_prediction('B (right of)', options) == 'B'
    assert SpatialMQA._parse_prediction('A, but right of', options) is None
    assert SpatialMQA._parse_prediction('A or B', options) is None


@pytest.mark.parametrize(
    'record,error',
    [
        (
            {
                'image': '000000000001.jpg',
                'question': 'Duplicate choices?',
                'options': ['left of', ' LEFT OF '],
                'answer': 'left of',
            },
            'unique',
        ),
        (
            {
                'image': '000000000001.jpg',
                'question': 'Missing gold?',
                'options': ['left of', 'right of'],
                'answer': 'behind',
            },
            'exactly one',
        ),
    ],
)
def test_rejects_non_unique_options_and_gold_outside_choices(tmp_path, record, error):
    image_root = tmp_path / 'images'
    write_image(image_root / record['image'])
    with pytest.raises(ValueError, match=error):
        SpatialMQA._records_to_dataframe([record], str(image_root))


def test_rejects_conflicting_anomalous_answer_column(tmp_path):
    image_root = tmp_path / 'images'
    write_image(image_root / '000000000001.jpg')
    record = make_records()[0] | {',answer': 'left of'}
    with pytest.raises(ValueError, match='Conflicting answer'):
        SpatialMQA._records_to_dataframe([record], str(image_root))


def test_exact_evaluation_joins_canonical_gold_and_ignores_forged_gold(tmp_path):
    dataset = make_dataset(tmp_path)
    prediction_file = tmp_path / 'predictions.csv'
    pd.DataFrame(
        [
            {
                'index': 'spatialmqa-test-000000',
                'prediction': 'right of',
                'answer': 'A',
                'options': json.dumps(['forged', 'gold']),
            },
            {
                'index': 'spatialmqa-test-000001',
                'prediction': 'A, but right of',
                'answer': 'A',
                'options': json.dumps(['forged', 'gold']),
            },
        ]
    ).to_csv(prediction_file, index=False)

    score = dataset.evaluate(str(prediction_file))
    assert score.loc[0, 'Overall'] == pytest.approx(50.0)
    assert score.loc[0, 'Overall_fraction'] == pytest.approx(0.5)
    assert score.loc[0, 'missing_predictions'] == 0
    assert score.loc[0, 'invalid_predictions'] == 1
    detail = pd.read_csv(tmp_path / 'predictions_spatialmqa_result.csv')
    assert detail['gold_answer'].tolist() == ['B', 'F']
    assert detail['hit'].tolist() == [1, 0]


def test_evaluation_rejects_duplicate_and_unknown_indices(tmp_path):
    dataset = make_dataset(tmp_path)
    duplicate_file = tmp_path / 'duplicate.csv'
    pd.DataFrame(
        [
            {'index': 'spatialmqa-test-000000', 'prediction': 'B'},
            {'index': 'spatialmqa-test-000000', 'prediction': 'B'},
        ]
    ).to_csv(duplicate_file, index=False)
    with pytest.raises(ValueError, match='duplicate indices'):
        dataset.evaluate(str(duplicate_file))

    unknown_file = tmp_path / 'unknown.csv'
    pd.DataFrame(
        [
            {'index': 'spatialmqa-test-000000', 'prediction': 'B'},
            {'index': 'spatialmqa-test-000001', 'prediction': 'F'},
            {'index': 'spatialmqa-test-999999', 'prediction': 'A'},
        ]
    ).to_csv(unknown_file, index=False)
    with pytest.raises(ValueError, match='unknown canonical indices'):
        dataset.evaluate(str(unknown_file))


def test_evaluation_keeps_missing_prediction_in_denominator(tmp_path):
    dataset = make_dataset(tmp_path)
    missing_file = tmp_path / 'missing.csv'
    pd.DataFrame(
        [{'index': 'spatialmqa-test-000000', 'prediction': 'B'}]
    ).to_csv(missing_file, index=False)

    score = dataset.evaluate(str(missing_file))
    assert score.loc[0, 'Overall'] == pytest.approx(50.0)
    assert score.loc[0, 'Overall_fraction'] == pytest.approx(0.5)
    assert score.loc[0, 'total'] == 2
    assert score.loc[0, 'missing_predictions'] == 1
    assert score.loc[0, 'invalid_predictions'] == 0
    detail = pd.read_csv(tmp_path / 'missing_spatialmqa_result.csv')
    assert detail['prediction_status'].tolist() == ['valid', 'missing']
    assert detail['hit'].tolist() == [1, 0]


def test_local_prepare_publishes_validated_manifest_and_detects_missing_image(tmp_path, monkeypatch):
    snapshot_root = tmp_path / 'snapshot'
    for record in make_records():
        write_image(snapshot_root / 'images' / record['image'])
    annotation_path = snapshot_root / SpatialMQA.HF_SPLIT_FILE
    annotation_path.write_text(
        ''.join(json.dumps(record) + '\n' for record in make_records()),
        encoding='utf-8',
    )
    monkeypatch.setattr(SpatialMQA, 'HF_SPLIT_SHA256', SpatialMQA._sha256(annotation_path))

    manifest = SpatialMQA._publish_snapshot(
        annotation_path=str(annotation_path),
        snapshot_root=str(snapshot_root),
        lmu_data_root=str(tmp_path / 'LMUData'),
        expected_rows=2,
    )
    prepared_root = tmp_path / 'LMUData' / SpatialMQA.PREPARED_DIRNAME
    data, validated_manifest = SpatialMQA.validate_prepared(
        str(prepared_root), expected_rows=2
    )
    assert manifest == validated_manifest
    assert len(data) == 2
    assert manifest['source']['revision'] == SpatialMQA.HF_REVISION
    assert manifest['source']['sha256'] == SpatialMQA.HF_SPLIT_SHA256
    assert manifest['checks']['all_image_paths_exist'] is True
    assert manifest['checks']['pil_verified_image_count'] == 2
    assert Path(data.loc[0, 'image_path']).is_file()

    image_path = Path(data.loc[0, 'image_path'])
    original_bytes = image_path.read_bytes()
    image_path.write_bytes(original_bytes + b'tampered')
    with pytest.raises(ValueError, match='size mismatch'):
        SpatialMQA.validate_prepared(str(prepared_root), expected_rows=2)
    image_path.write_bytes(original_bytes)
    image_path.unlink()
    with pytest.raises(FileNotFoundError, match='image is missing'):
        SpatialMQA.validate_prepared(str(prepared_root), expected_rows=2)
