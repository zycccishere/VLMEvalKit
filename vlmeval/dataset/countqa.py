import hashlib
import json
import numbers
import os
import re
from pathlib import Path

import pandas as pd

from .image_base import ImageBaseDataset
from ..smp.file import LMUDataRoot


COUNTQA_DATASET_ID = 'Jayant-Sravan/CountQA'
COUNTQA_REVISION = 'f92cc6fe46542c61e2916e3d2ae9a911e2216b1a'
COUNTQA_SPLIT = 'test'
COUNTQA_IMAGE_ROWS = 1001
COUNTQA_QA_ROWS = 1528
COUNTQA_NAME = 'CountQA'

_INTEGER_PATTERN = re.compile(r'[0-9]+', flags=re.ASCII)
_PRESERVED_FIELDS = ('objects', 'categories', 'is_focused')
_REQUIRED_TABLE_FIELDS = (
    'index',
    'source_image_row',
    'question_ordinal',
    'question',
    'answer',
    'image_path',
    *_PRESERVED_FIELDS,
)


def _non_negative_ordinal(value, field):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f'{field} must be a non-negative integer, got {value!r}.')
    return int(value)


def make_countqa_index(image_row, question_ordinal):
    """Build the stable, zero-based annotation key used for inference and scoring."""
    image_row = _non_negative_ordinal(image_row, 'image_row')
    question_ordinal = _non_negative_ordinal(question_ordinal, 'question_ordinal')
    return f'countqa-{image_row:04d}-{question_ordinal:02d}'


def parse_countqa_integer(value):
    """Parse one complete non-negative integer response, without extracting substrings."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value) if value >= 0 else None
    if isinstance(value, numbers.Real):
        # Model generations are text. Accepting 2.0 here would make the metric
        # depend on CSV/XLSX dtype inference instead of the emitted string.
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if _INTEGER_PATTERN.fullmatch(text) is None:
        return None
    return int(text)


def _require_list(row, field, image_row):
    value = row.get(field)
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f'CountQA image row {image_row} field {field!r} must be a list, '
            f'got {type(value).__name__}.'
        )
    return list(value)


def expand_countqa_row(row, image_row, image_path=None):
    """Expand one image-level source row into canonical annotation-level records."""
    image_row = _non_negative_ordinal(image_row, 'image_row')
    questions = _require_list(row, 'questions', image_row)
    answers = _require_list(row, 'answers', image_row)
    if len(questions) != len(answers):
        raise ValueError(
            f'CountQA image row {image_row} has {len(questions)} questions but '
            f'{len(answers)} answers; refusing positional truncation.'
        )

    missing_metadata = [field for field in _PRESERVED_FIELDS if field not in row]
    if missing_metadata:
        raise KeyError(
            f'CountQA image row {image_row} is missing metadata fields: '
            f'{", ".join(missing_metadata)}.'
        )

    for field in ('objects', 'categories'):
        if not isinstance(row[field], (list, tuple)):
            raise TypeError(
                f'CountQA image row {image_row} field {field!r} must be a list, '
                f'got {type(row[field]).__name__}.'
            )

    if image_path is None:
        image_path = row.get('image_path')
    records = []
    for question_ordinal, (question, answer) in enumerate(zip(questions, answers)):
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f'CountQA image row {image_row} question {question_ordinal} is empty or not text.'
            )
        answer_integer = parse_countqa_integer(answer)
        if answer_integer is None:
            raise ValueError(
                f'CountQA image row {image_row} answer {question_ordinal} is not a '
                f'non-negative integer: {answer!r}.'
            )
        record = {
            'index': make_countqa_index(image_row, question_ordinal),
            'source_image_row': image_row,
            'question_ordinal': question_ordinal,
            'question': question.strip(),
            'answer': str(answer_integer),
            'objects': list(row['objects']),
            'categories': list(row['categories']),
            'is_focused': bool(row['is_focused']),
        }
        if image_path is not None:
            record['image_path'] = os.fspath(image_path)
        records.append(record)
    return records


def expand_countqa_rows(rows):
    records = []
    for image_row, row in enumerate(rows):
        records.extend(expand_countqa_row(row, image_row))
    return records


def _canonical_index(value, context):
    if value is None:
        raise ValueError(f'{context} contains an empty index.')
    try:
        is_missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        is_missing = False
    if is_missing:
        raise ValueError(f'{context} contains an empty index.')
    index = str(value).strip()
    if not index:
        raise ValueError(f'{context} contains an empty index.')
    return index


def _duplicate_indices(data, context):
    duplicated = data.loc[data['index'].duplicated(keep=False), 'index'].unique().tolist()
    if duplicated:
        preview = ', '.join(str(value) for value in duplicated[:10])
        raise ValueError(f'{context} contains duplicate canonical index values: {preview}.')


def score_countqa_predictions(annotations, predictions):
    """Align predictions by canonical index and score every source annotation exactly once."""
    if not isinstance(annotations, pd.DataFrame):
        annotations = pd.DataFrame(annotations)
    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(predictions)
    for field in ('index', 'answer'):
        if field not in annotations:
            raise KeyError(f'CountQA annotations are missing required column {field!r}.')
    for field in ('index', 'prediction'):
        if field not in predictions:
            raise KeyError(f'CountQA predictions are missing required column {field!r}.')

    annotations = annotations.copy().reset_index(drop=True)
    predictions = predictions[['index', 'prediction']].copy().reset_index(drop=True)
    annotations['index'] = [
        _canonical_index(value, 'CountQA annotations') for value in annotations['index']
    ]
    predictions['index'] = [
        _canonical_index(value, 'CountQA predictions') for value in predictions['index']
    ]
    _duplicate_indices(annotations, 'CountQA annotations')
    _duplicate_indices(predictions, 'CountQA predictions')

    annotation_indices = set(annotations['index'])
    unexpected = sorted(set(predictions['index']) - annotation_indices)
    if unexpected:
        preview = ', '.join(unexpected[:10])
        raise ValueError(f'CountQA predictions contain unknown canonical index values: {preview}.')

    prediction_map = dict(zip(predictions['index'], predictions['prediction']))
    details = annotations.copy()
    raw_predictions = []
    parsed_predictions = []
    parse_statuses = []
    exact_scores = []
    answer_integers = []
    for _, annotation in annotations.iterrows():
        index = annotation['index']
        answer_integer = parse_countqa_integer(annotation['answer'])
        if answer_integer is None:
            raise ValueError(
                f'CountQA annotation {index} has invalid integer answer {annotation["answer"]!r}.'
            )

        if index not in prediction_map:
            raw_prediction = None
            parsed_prediction = None
            parse_status = 'missing'
        else:
            raw_prediction = prediction_map[index]
            parsed_prediction = parse_countqa_integer(raw_prediction)
            parse_status = 'valid' if parsed_prediction is not None else 'invalid_format'

        raw_predictions.append(raw_prediction)
        parsed_predictions.append(parsed_prediction)
        parse_statuses.append(parse_status)
        answer_integers.append(answer_integer)
        exact_scores.append(
            int(parsed_prediction == answer_integer) if parsed_prediction is not None else 0
        )

    details['answer_integer'] = answer_integers
    details['prediction'] = raw_predictions
    details['prediction_integer'] = parsed_predictions
    details['parse_status'] = parse_statuses
    details['exact_match'] = exact_scores

    annotation_count = len(details)
    correct_count = int(details['exact_match'].sum())
    missing_count = int((details['parse_status'] == 'missing').sum())
    invalid_count = int((details['parse_status'] == 'invalid_format').sum())
    accuracy = correct_count / annotation_count if annotation_count else 0.0
    summary = pd.DataFrame([{
        'metric': 'annotation_exact_accuracy',
        'score': accuracy,
        'accuracy_percent': accuracy * 100.0,
        'correct': correct_count,
        'annotations': annotation_count,
        'predictions': len(predictions),
        'missing': missing_count,
        'invalid_format': invalid_count,
    }])
    return details, summary


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_list_metadata(value, field, index):
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        raise ValueError(f'CountQA row {index} field {field!r} is not a JSON list.')
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f'CountQA row {index} field {field!r} is not valid JSON.') from exc
    if not isinstance(decoded, list):
        raise ValueError(f'CountQA row {index} field {field!r} is not a JSON list.')
    return decoded


def _read_local_table(path):
    suffix = path.suffix.lower()
    if suffix == '.tsv':
        data = pd.read_csv(
            path,
            sep='\t',
            dtype={'index': str, 'answer': str, 'image_path': str},
        )
    elif suffix == '.csv':
        data = pd.read_csv(path, dtype={'index': str, 'answer': str, 'image_path': str})
    else:
        raise ValueError(f'CountQA only supports prepared TSV/CSV files, got {path}.')
    return data


def _resolve_under_root(root, relative_path, label):
    root = Path(root).resolve()
    candidate = Path(str(relative_path))
    if candidate.is_absolute():
        raise ValueError(f'CountQA {label} path must be relative to LMUData: {relative_path!r}.')
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'CountQA {label} path escapes LMUData: {relative_path!r}.') from exc
    return resolved


class CountQADataset(ImageBaseDataset):
    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    FORCE_DATASET_PROMPT = True
    DATASET_URL = {COUNTQA_NAME: ''}

    @classmethod
    def supported_datasets(cls):
        return [COUNTQA_NAME]

    def load_data(self, dataset):
        if dataset != COUNTQA_NAME:
            raise ValueError(f'Unsupported CountQA dataset name: {dataset!r}.')

        data_root = Path(LMUDataRoot())
        manifest_path = data_root / f'{COUNTQA_NAME}.manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f'CountQA manifest not found at {manifest_path}; revision and checksum '
                'cannot be verified. Run `python scripts/prepare_countqa.py '
                '--lmu-data <LMUData>` in the Qwen environment.'
            )
        with open(manifest_path, encoding='utf-8') as stream:
            manifest = json.load(stream)
        if manifest.get('dataset_id') != COUNTQA_DATASET_ID:
            raise ValueError(f'Unexpected CountQA manifest dataset_id in {manifest_path}.')
        if manifest.get('revision') != COUNTQA_REVISION:
            raise ValueError(
                f'CountQA manifest revision is {manifest.get("revision")!r}; '
                f'expected pinned revision {COUNTQA_REVISION}.'
            )
        expected_counts = {'image_rows': COUNTQA_IMAGE_ROWS, 'qa_rows': COUNTQA_QA_ROWS}
        if manifest.get('counts') != expected_counts:
            raise ValueError(f'Unexpected CountQA counts in {manifest_path}.')
        recorded_names = [entry.get('path') for entry in manifest.get('tables', [])]
        candidates = [
            data_root / name
            for name in (f'{COUNTQA_NAME}.tsv', f'{COUNTQA_NAME}.csv')
            if name in recorded_names
        ]

        data_path = next((path for path in candidates if path.is_file()), None)
        if data_path is None:
            raise FileNotFoundError(
                f'Prepared CountQA table not found under {data_root}. Run '
                '`python scripts/prepare_countqa.py --lmu-data <LMUData>` in the Qwen '
                'environment.'
            )

        table_entry = next(
            (
                entry
                for entry in manifest.get('tables', [])
                if entry.get('path') == data_path.name
            ),
            None,
        )
        if table_entry is None:
            raise ValueError(f'{data_path.name} is not recorded in {manifest_path}.')
        actual_checksum = _sha256_file(data_path)
        if actual_checksum != table_entry.get('sha256'):
            raise ValueError(f'CountQA table checksum mismatch for {data_path}.')

        image_entries = manifest.get('images', [])
        if len(image_entries) != COUNTQA_IMAGE_ROWS:
            raise ValueError(
                f'CountQA manifest records {len(image_entries)} images; '
                f'expected {COUNTQA_IMAGE_ROWS}.'
            )
        manifest_image_paths = set()
        for entry in image_entries:
            image_path = _resolve_under_root(data_root, entry.get('path', ''), 'manifest image')
            manifest_image_paths.add(image_path)
            if not image_path.is_file():
                raise FileNotFoundError(f'CountQA prepared image is missing: {image_path}.')
            if image_path.stat().st_size != entry.get('bytes'):
                raise ValueError(f'CountQA prepared image size mismatch: {image_path}.')
            if _sha256_file(image_path) != entry.get('sha256'):
                raise ValueError(f'CountQA prepared image checksum mismatch: {image_path}.')

        data = _read_local_table(data_path)
        missing_fields = [field for field in _REQUIRED_TABLE_FIELDS if field not in data]
        if missing_fields:
            raise ValueError(
                f'Prepared CountQA table is missing columns: {", ".join(missing_fields)}.'
            )
        if len(data) != COUNTQA_QA_ROWS:
            raise ValueError(
                f'Prepared CountQA table has {len(data)} rows; expected {COUNTQA_QA_ROWS}.'
            )
        if data['source_image_row'].nunique() != COUNTQA_IMAGE_ROWS:
            raise ValueError(
                'Prepared CountQA table does not contain exactly '
                f'{COUNTQA_IMAGE_ROWS} source images.'
            )

        data['index'] = [str(value) for value in data['index']]
        _duplicate_indices(data, 'Prepared CountQA table')
        expected_indices = [
            make_countqa_index(image_row, question_ordinal)
            for image_row, question_ordinal in zip(
                data['source_image_row'], data['question_ordinal']
            )
        ]
        if data['index'].tolist() != expected_indices:
            raise ValueError(
                'Prepared CountQA table contains non-canonical or reordered index fields.'
            )
        for field in ('objects', 'categories'):
            data[field] = [
                _decode_list_metadata(value, field, index)
                for value, index in zip(data[field], data['index'])
            ]
        table_image_paths = {
            _resolve_under_root(
                data_root,
                Path('images') / COUNTQA_NAME / str(path),
                'table image',
            )
            for path in data['image_path'].unique()
        }
        if table_image_paths != manifest_image_paths:
            raise ValueError('CountQA table images do not match the manifest inventory.')

        self.data_path = os.fspath(data_path)
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_path = line['image_path']
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError(f'CountQA row {line.get("index", "<unknown>")} has no image_path.')
        image_path = Path(image_path)
        if not image_path.is_absolute():
            image_path = Path(self.img_root) / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f'CountQA image not found: {image_path}.')

        question = line['question']
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f'CountQA row {line.get("index", "<unknown>")} has no question.')
        prompt = (
            f'{question.strip()}\n'
            'Answer with exactly one non-negative integer (0 or greater). '
            'Output the integer only; do not include words, punctuation, or an explanation.'
        )
        return [
            {'type': 'image', 'value': os.fspath(image_path)},
            {'type': 'text', 'value': prompt},
        ]

    def evaluate(self, eval_file, **judge_kwargs):
        del judge_kwargs
        eval_path = Path(eval_file)
        suffix = eval_path.suffix.lower()
        if suffix == '.tsv':
            predictions = pd.read_csv(
                eval_path,
                sep='\t',
                dtype={'index': str, 'prediction': str},
                keep_default_na=False,
            )
        elif suffix == '.csv':
            predictions = pd.read_csv(
                eval_path,
                dtype={'index': str, 'prediction': str},
                keep_default_na=False,
            )
        elif suffix in ('.xlsx', '.xls'):
            predictions = pd.read_excel(
                eval_path,
                dtype={'index': str, 'prediction': str},
                keep_default_na=False,
            )
        else:
            raise ValueError(f'Unsupported CountQA prediction format: {suffix}.')

        details, summary = score_countqa_predictions(self.data, predictions)
        stem = eval_path.with_suffix('')
        details.to_csv(f'{stem}_countqa_details.csv', index=False)
        summary.to_csv(f'{stem}_acc.csv', index=False)
        return summary


CountQA = CountQADataset
