import numbers
import re
from pathlib import Path

import pandas as pd

from .image_base import ImageBaseDataset
from ..smp.file import load


COUNTQA_NAME = 'CountQA'
COUNTQA_POST_PROMPT = ' Please answer with only a number.'
_INTEGER_PATTERN = re.compile(r'[-+]?\d+', flags=re.ASCII)
_NUMBER_WORDS = {
    word: value for value, word in enumerate((
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
        'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
        'seventeen', 'eighteen', 'nineteen', 'twenty',
    ))
}


def _extract_final_answer(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if '<answer>' in text and '</answer>' in text:
        text = text.split('<answer>', 1)[1].split('</answer>', 1)[0].strip()
    if '</think>' in text:
        text = text.split('</think>', 1)[-1].strip()
    boxed = re.findall(r'\\boxed\{([^{}]*)\}', text)
    return boxed[-1].strip() if boxed else text


def parse_countqa_integer(value):
    # Mirrors zlab-princeton/vero countqa_process_results at c37e1284.
    if isinstance(value, numbers.Real):
        try:
            return int(value)
        except (OverflowError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    text = str(_extract_final_answer(value)).strip().lower()
    match = _INTEGER_PATTERN.search(text)
    if match:
        return int(match.group(0))
    for word, number in _NUMBER_WORDS.items():
        if word in text:
            return number
    return None


def _canonical_indices(data, label):
    if 'index' not in data:
        raise KeyError(f'{label} is missing the index column.')
    indices = data['index'].astype(str).str.strip()
    if indices.eq('').any() or indices.duplicated().any():
        raise ValueError(f'{label} contains empty or duplicate indices.')
    return indices


def score_countqa_predictions(annotations, predictions):
    annotations = pd.DataFrame(annotations).copy().reset_index(drop=True)
    predictions = pd.DataFrame(predictions).copy().reset_index(drop=True)
    if 'answer' not in annotations or 'prediction' not in predictions:
        raise KeyError('CountQA scoring requires answer and prediction columns.')
    annotations['index'] = _canonical_indices(annotations, 'CountQA annotations')
    predictions['index'] = _canonical_indices(predictions, 'CountQA predictions')
    unknown = sorted(set(predictions['index']) - set(annotations['index']))
    if unknown:
        raise ValueError(f'CountQA predictions contain unknown indices: {unknown[:5]}')

    prediction_map = dict(zip(predictions['index'], predictions['prediction']))
    details = annotations.copy()
    details['prediction'] = [prediction_map.get(index) for index in details['index']]
    details['answer_integer'] = [parse_countqa_integer(value) for value in details['answer']]
    if any(value is None for value in details['answer_integer']):
        raise ValueError('CountQA annotations contain a non-integer answer.')
    prediction_integers = [
        parse_countqa_integer(value) for value in details['prediction']
    ]
    details['prediction_integer'] = prediction_integers
    details['parse_status'] = [
        'missing' if index not in prediction_map else
        ('valid' if parsed is not None else 'invalid_format')
        for index, parsed in zip(details['index'], prediction_integers)
    ]
    details['exact_match'] = [
        int(prediction == answer) if prediction is not None else 0
        for prediction, answer in zip(prediction_integers, details['answer_integer'])
    ]

    total = len(details)
    correct = int(details['exact_match'].sum())
    accuracy = correct / total if total else 0.0
    summary = pd.DataFrame([{
        'Overall': accuracy,
        'accuracy_percent': accuracy * 100.0,
        'correct': correct,
        'annotations': total,
        'predictions': len(predictions),
        'missing': int((details['parse_status'] == 'missing').sum()),
        'invalid_format': int((details['parse_status'] == 'invalid_format').sum()),
    }])
    return details, summary


def _read_predictions(path):
    path = Path(path)
    kwargs = {'dtype': {'index': str, 'prediction': str}, 'keep_default_na': False}
    if path.suffix.lower() == '.tsv':
        return pd.read_csv(path, sep='\t', **kwargs)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path, **kwargs)
    if path.suffix.lower() in ('.xlsx', '.xls'):
        return pd.read_excel(path, **kwargs)
    return load(str(path))


class CountQADataset(ImageBaseDataset):
    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    FORCE_DATASET_PROMPT = True
    DATASET_URL = {COUNTQA_NAME: ''}

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        images = self.dump_image(line)
        prompt = f'{str(line["question"]).strip()}{COUNTQA_POST_PROMPT}'
        return [
            *[{'type': 'image', 'value': image} for image in images],
            {'type': 'text', 'value': prompt},
        ]

    def evaluate(self, eval_file, **judge_kwargs):
        del judge_kwargs
        path = Path(eval_file)
        details, summary = score_countqa_predictions(self.data, _read_predictions(path))
        details.to_csv(path.with_name(f'{path.stem}_detail.csv'), index=False)
        summary.to_csv(path.with_name(f'{path.stem}_acc.csv'), index=False)
        return summary


CountQA = CountQADataset
