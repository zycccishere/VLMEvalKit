import os
from pathlib import Path

import pandas as pd
from PIL import Image

from vlmeval.dataset.countqa import (
    CountQADataset,
    parse_countqa_integer,
    score_countqa_predictions,
)


def test_public_toolkit_first_number_parser():
    assert parse_countqa_integer('0') == 0
    assert parse_countqa_integer('400') == 400
    assert parse_countqa_integer('There are 2.') == 2
    assert parse_countqa_integer('2 or 3') == 2
    assert parse_countqa_integer('two') == 2
    assert parse_countqa_integer('<think>1 or 2</think><answer>3</answer>') == 3
    assert parse_countqa_integer(r'The final count is \boxed{12}.') == 12
    for value in ('', 'many', None):
        assert parse_countqa_integer(value) is None


def test_score_keeps_missing_and_invalid_predictions_in_denominator():
    annotations = pd.DataFrame([
        {'index': 'countqa-0000-00', 'answer': '0'},
        {'index': 'countqa-0001-00', 'answer': '400'},
        {'index': 'countqa-0002-00', 'answer': '2'},
    ])
    predictions = pd.DataFrame([
        {'index': 'countqa-0001-00', 'prediction': '400'},
        {'index': 'countqa-0002-00', 'prediction': 'many'},
    ])
    details, summary = score_countqa_predictions(annotations, predictions)
    assert details['exact_match'].tolist() == [0, 1, 0]
    assert summary.loc[0, 'Overall'] == 1 / 3
    assert summary.loc[0, 'missing'] == 1
    assert summary.loc[0, 'invalid_format'] == 1


def test_thin_loader_and_dataset_owned_prompt(tmp_path, monkeypatch):
    image_root = tmp_path / 'images' / 'CountQA'
    image_root.mkdir(parents=True)
    image = image_root / 'countqa-0000.jpg'
    Image.new('RGB', (4, 3)).save(image)
    pd.DataFrame([{
        'index': 'countqa-0000-00',
        'source_image_row': 0,
        'question_ordinal': 0,
        'image_path': image.name,
        'question': 'How many objects?',
        'answer': '2',
    }]).to_csv(tmp_path / 'CountQA.tsv', sep='\t', index=False)
    monkeypatch.setenv('LMUData', str(tmp_path))

    dataset = CountQADataset(dataset='CountQA')
    prompt = dataset.build_prompt(0)
    assert dataset.FORCE_DATASET_PROMPT is True
    assert [item['type'] for item in prompt] == ['image', 'text']
    assert prompt[0]['value'] == os.fspath(image)
    assert prompt[1]['value'] == 'How many objects? Please answer with only a number.'


def test_evaluate_uses_first_number_rewrite_semantics(tmp_path):
    dataset = object.__new__(CountQADataset)
    dataset.data = pd.DataFrame([{'index': 'countqa-0000-00', 'answer': '2'}])
    prediction_file = tmp_path / 'predictions.csv'
    pd.DataFrame([{
        'index': 'countqa-0000-00',
        'prediction': '2.0',
    }]).to_csv(prediction_file, index=False)
    summary = dataset.evaluate(prediction_file)
    assert summary.loc[0, 'Overall'] == 1.0
    assert summary.loc[0, 'invalid_format'] == 0
