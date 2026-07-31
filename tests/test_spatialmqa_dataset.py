import importlib.util
import os
from pathlib import Path

import pandas as pd
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_spatialmqa.py'
SPEC = importlib.util.spec_from_file_location('prepare_spatialmqa', SCRIPT)
prepare_spatialmqa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_spatialmqa)


def test_prepare_converts_text_answers_to_standard_mcq_rows(tmp_path):
    image_root = tmp_path / 'images'
    image_root.mkdir()
    Image.new('RGB', (4, 3)).save(image_root / '000000000001.jpg')
    records = [{
        'image': '000000000001.jpg',
        'question': 'Where is the object?',
        'options': ['left of', 'right of'],
        'answer': 'right of',
    }]
    data = prepare_spatialmqa.records_to_dataframe(records, image_root)
    assert data.loc[0, ['A', 'B', 'answer', 'relation']].tolist() == [
        'left of', 'right of', 'B', 'right of'
    ]
    assert data.loc[0, 'index'] == 'spatialmqa-test-000000'


def test_official_spatialmqa_prompt_uses_relation_text_options(tmp_path, monkeypatch):
    from vlmeval.dataset.spatialmqa import SpatialMQADataset

    image = tmp_path / 'spatial.jpg'
    Image.new('RGB', (4, 3)).save(image)
    pd.DataFrame([{
        'index': 'spatialmqa-test-000000',
        'image_path': str(image),
        'question': 'Where is the object?',
        'A': 'left of',
        'B': 'right of',
        'answer': 'B',
        'relation': 'right of',
    }]).to_csv(tmp_path / 'SpatialMQA.tsv', sep='\t', index=False)
    monkeypatch.setenv('LMUData', str(tmp_path))

    dataset = SpatialMQADataset(dataset='SpatialMQA')
    prompt = dataset.build_prompt(0)
    assert dataset.FORCE_DATASET_PROMPT is True
    assert [item['type'] for item in prompt] == ['image', 'text']
    assert 'Input: Image: provided above' in prompt[1]['value']
    assert '<image>' not in prompt[1]['value']
    assert 'Options: left of; right of.' in prompt[1]['value']
    assert 'without explaining any reason' in prompt[1]['value']


def test_official_spatialmqa_metric_scores_relation_text_not_choice_letter(tmp_path):
    from vlmeval.dataset.spatialmqa import SpatialMQADataset

    dataset = object.__new__(SpatialMQADataset)
    dataset.data = pd.DataFrame([{
        'index': 'spatialmqa-test-000000',
        'A': 'left of',
        'B': 'right of',
        'answer': 'B',
        'relation': 'right of',
    }])
    prediction_file = tmp_path / 'predictions.csv'
    pd.DataFrame([
        {'index': 'spatialmqa-test-000000', 'prediction': 'The answer is right of.'},
    ]).to_csv(prediction_file, index=False)
    summary = dataset.evaluate(prediction_file)
    assert summary.loc[0, 'Overall'] == 1.0

    pd.DataFrame([
        {'index': 'spatialmqa-test-000000', 'prediction': 'B'},
    ]).to_csv(prediction_file, index=False)
    summary = dataset.evaluate(prediction_file)
    assert summary.loc[0, 'Overall'] == 0.0


def test_spatialmqa_missing_predictions_remain_in_denominator(tmp_path):
    from vlmeval.dataset.spatialmqa import SpatialMQADataset

    dataset = object.__new__(SpatialMQADataset)
    dataset.data = pd.DataFrame([
        {
            'index': 'spatialmqa-test-000000',
            'A': 'left of',
            'B': 'right of',
            'answer': 'B',
            'relation': 'right of',
        },
        {
            'index': 'spatialmqa-test-000001',
            'A': 'behind',
            'B': 'in front of',
            'answer': 'A',
            'relation': 'behind',
        },
    ])
    prediction_file = tmp_path / 'predictions.csv'
    pd.DataFrame([{
        'index': 'spatialmqa-test-000000',
        'prediction': 'right of',
    }]).to_csv(prediction_file, index=False)

    summary = dataset.evaluate(prediction_file)
    assert summary.loc[0, 'Overall'] == 0.5
    assert summary.loc[0, 'total'] == 2
    assert summary.loc[0, 'predictions'] == 1
    assert summary.loc[0, 'missing'] == 1
