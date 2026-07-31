#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import string
from pathlib import Path

import pandas as pd


REPO_ID = 'liuziyan/SpatialMQA'
REVISION = '2c297135743209b91fe0729c033b49bb1d72f788'
SPLIT_FILE = 'test.jsonl'
SPLIT_SHA256 = 'f5e3a76059087dba72b9f9396981f0c3df128b6335f9aa12655ea580b691ba9e'
EXPECTED_ROWS = 1076
EXPECTED_IMAGES = 1066


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_answer(options, value):
    text = str(value).strip()
    if len(text) == 1 and text.upper() in string.ascii_uppercase[:len(options)]:
        return text.upper()
    matches = [i for i, option in enumerate(options) if str(option).strip().casefold() == text.casefold()]
    if len(matches) != 1:
        raise ValueError(f'Gold answer {value!r} does not identify exactly one option.')
    return string.ascii_uppercase[matches[0]]


def records_to_dataframe(records, image_root):
    rows = []
    image_root = Path(image_root).resolve()
    for source_row, record in enumerate(records):
        options = record.get('options')
        if not isinstance(options, list) or not 2 <= len(options) <= 6:
            raise ValueError(f'SpatialMQA row {source_row} must contain 2-6 options.')
        if len({str(option).strip().casefold() for option in options}) != len(options):
            raise ValueError(f'SpatialMQA row {source_row} contains duplicate options.')
        answer = canonical_answer(options, record.get('answer'))
        alternate = record.get(',answer')
        if alternate not in (None, '') and canonical_answer(options, alternate) != answer:
            raise ValueError(f'SpatialMQA row {source_row} has conflicting answer fields.')
        image_path = image_root / str(record.get('image', '')).strip()
        if not image_path.is_file():
            raise FileNotFoundError(f'SpatialMQA image is missing: {image_path}')
        row = {
            'index': f'spatialmqa-test-{source_row:06d}',
            'image_path': os.fspath(image_path),
            'question': str(record.get('question', '')).strip(),
            'answer': answer,
            'relation': str(options[ord(answer) - ord('A')]).strip(),
        }
        if not row['question']:
            raise ValueError(f'SpatialMQA row {source_row} has an empty question.')
        row.update({letter: option for letter, option in zip(string.ascii_uppercase, options)})
        rows.append(row)
    return pd.DataFrame(rows)


def validate_table(path, expected_rows=EXPECTED_ROWS, expected_images=EXPECTED_IMAGES):
    data = pd.read_csv(path, sep='\t', dtype={'index': str, 'answer': str})
    required = {'index', 'image_path', 'question', 'A', 'B', 'answer', 'relation'}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f'SpatialMQA TSV is missing columns: {missing}')
    if len(data) != expected_rows:
        raise ValueError(f'SpatialMQA TSV has {len(data)} rows, expected {expected_rows}.')
    if data['index'].duplicated().any():
        raise ValueError('SpatialMQA TSV contains duplicate indices.')
    missing_images = [path for path in data['image_path'].unique() if not Path(path).is_file()]
    if missing_images:
        raise FileNotFoundError(f'SpatialMQA TSV references a missing image: {missing_images[0]}')
    if data['image_path'].nunique() != expected_images:
        raise ValueError(
            f'SpatialMQA TSV has {data["image_path"].nunique()} unique images, '
            f'expected {expected_images}.'
        )
    for row in data.to_dict('records'):
        answer = str(row['answer']).strip().upper()
        if answer not in string.ascii_uppercase or pd.isna(row.get(answer)):
            raise ValueError(f'SpatialMQA row {row["index"]} has invalid answer {answer!r}.')
        if str(row[answer]).strip().casefold() != str(row['relation']).strip().casefold():
            raise ValueError(f'SpatialMQA row {row["index"]} has inconsistent relation text.')
    return data


def publish_existing(source, target):
    source = Path(source).resolve()
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f'.{target.name}.tmp')
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(os.path.relpath(source, target.parent))
    os.replace(temporary, target)


def prepare(lmu_data, source_tsv=None, cache_dir=None):
    lmu_data = Path(lmu_data).expanduser().resolve()
    target = lmu_data / 'SpatialMQA.tsv'
    if source_tsv is None:
        candidates = sorted((lmu_data / 'SpatialMQA' / 'revisions').glob('*/SpatialMQA.tsv'))
        source_tsv = candidates[-1] if candidates else None
    if source_tsv is not None:
        validate_table(source_tsv)
        publish_existing(source_tsv, target)
        return validate_table(target)

    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        repo_id=REPO_ID,
        repo_type='dataset',
        revision=REVISION,
        cache_dir=cache_dir,
    ))
    source = snapshot / SPLIT_FILE
    if sha256(source) != SPLIT_SHA256:
        raise ValueError('SpatialMQA source checksum does not match the pinned test split.')
    records = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line]
    image_root = lmu_data / 'images' / 'SpatialMQA'
    image_root.mkdir(parents=True, exist_ok=True)
    for image_name in {str(record['image']).strip() for record in records}:
        destination = image_root / image_name
        if not destination.is_file():
            shutil.copy2(snapshot / 'images' / image_name, destination)
    data = records_to_dataframe(records, image_root)
    data.to_csv(target, sep='\t', index=False)
    return validate_table(target)


def main():
    parser = argparse.ArgumentParser(description='Prepare SpatialMQA as a standard VLMEvalKit MCQ TSV.')
    parser.add_argument('--lmu-data', default=os.environ.get('LMUData', str(Path.home() / 'LMUData')))
    parser.add_argument('--source-tsv')
    parser.add_argument('--cache-dir')
    args = parser.parse_args()
    data = prepare(args.lmu_data, source_tsv=args.source_tsv, cache_dir=args.cache_dir)
    print(json.dumps({'rows': len(data), 'images': data['image_path'].nunique()}, sort_keys=True))


if __name__ == '__main__':
    main()
