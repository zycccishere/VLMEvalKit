#!/usr/bin/env python3
import argparse
import hashlib
import io
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import PIL
from PIL import Image


COUNTQA_DATASET_ID = 'Jayant-Sravan/CountQA'
COUNTQA_REVISION = 'f92cc6fe46542c61e2916e3d2ae9a911e2216b1a'
COUNTQA_SPLIT = 'test'
COUNTQA_IMAGE_ROWS = 1001
COUNTQA_QA_ROWS = 1528
COUNTQA_NAME = 'CountQA'


def make_countqa_index(image_row, question_ordinal):
    if image_row < 0 or question_ordinal < 0:
        raise ValueError('CountQA index components must be non-negative.')
    return f'countqa-{image_row:04d}-{question_ordinal:02d}'


def expand_countqa_row(row, image_row, image_path):
    questions = row.get('questions')
    answers = row.get('answers')
    if not isinstance(questions, list) or not isinstance(answers, list):
        raise TypeError(f'CountQA image row {image_row} questions/answers must be lists.')
    if len(questions) != len(answers):
        raise ValueError(
            f'CountQA image row {image_row} has {len(questions)} questions but '
            f'{len(answers)} answers.'
        )
    for field in ('objects', 'categories'):
        if not isinstance(row.get(field), list):
            raise TypeError(f'CountQA image row {image_row} field {field!r} must be a list.')
    if 'is_focused' not in row:
        raise KeyError(f'CountQA image row {image_row} is missing is_focused.')

    records = []
    for question_ordinal, (question, answer) in enumerate(zip(questions, answers)):
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f'CountQA image row {image_row} question {question_ordinal} is not text.'
            )
        answer = str(answer).strip()
        if not answer.isascii() or not answer.isdigit():
            raise ValueError(
                f'CountQA image row {image_row} answer {question_ordinal} is not an integer.'
            )
        records.append({
            'index': make_countqa_index(image_row, question_ordinal),
            'source_image_row': image_row,
            'question_ordinal': question_ordinal,
            'question': question.strip(),
            'answer': str(int(answer)),
            'image_path': image_path,
            'objects': list(row['objects']),
            'categories': list(row['categories']),
            'is_focused': bool(row['is_focused']),
        })
    return records


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    with open(temporary, 'wb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_json(path, content):
    payload = json.dumps(content, indent=2, ensure_ascii=False, sort_keys=True).encode('utf-8')
    atomic_write_bytes(path, payload + b'\n')


def image_bytes_and_extension(image_value, image_row):
    if not isinstance(image_value, dict):
        raise TypeError(f'CountQA image row {image_row} has unexpected image payload type.')
    content = image_value.get('bytes')
    if not isinstance(content, (bytes, bytearray)):
        raise ValueError(f'CountQA image row {image_row} does not contain embedded image bytes.')
    content = bytes(content)
    with Image.open(io.BytesIO(content)) as image:
        image_format = (image.format or '').upper()
        image.verify()
    extensions = {'JPEG': 'jpg', 'PNG': 'png', 'WEBP': 'webp', 'TIFF': 'tif'}
    if image_format not in extensions:
        raise ValueError(
            f'CountQA image row {image_row} uses unsupported format {image_format!r}.'
        )
    return content, extensions[image_format]


def serialize_metadata(data):
    data = data.copy()
    for field in ('objects', 'categories'):
        data[field] = [
            json.dumps(value, ensure_ascii=False, separators=(',', ':')) for value in data[field]
        ]
    return data


def write_table(data, path):
    temporary = path.with_name(f'.{path.name}.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.tsv':
        data.to_csv(temporary, sep='\t', index=False)
    elif path.suffix == '.csv':
        data.to_csv(temporary, index=False)
    else:
        raise ValueError(f'Unsupported CountQA table format: {path.suffix}.')
    os.replace(temporary, path)


def read_table(path):
    if path.suffix == '.tsv':
        return pd.read_csv(
            path,
            sep='\t',
            dtype={'index': str, 'answer': str, 'image_path': str},
        )
    return pd.read_csv(path, dtype={'index': str, 'answer': str, 'image_path': str})


def verify_prepared_dataset(lmu_data, manifest_path=None):
    lmu_data = Path(lmu_data)
    manifest_path = manifest_path or lmu_data / f'{COUNTQA_NAME}.manifest.json'
    with open(manifest_path, encoding='utf-8') as stream:
        manifest = json.load(stream)

    if manifest.get('dataset_id') != COUNTQA_DATASET_ID:
        raise ValueError('Manifest dataset_id does not identify CountQA.')
    if manifest.get('revision') != COUNTQA_REVISION:
        raise ValueError('Manifest revision does not match the pinned CountQA revision.')
    expected_counts = {'image_rows': COUNTQA_IMAGE_ROWS, 'qa_rows': COUNTQA_QA_ROWS}
    if manifest.get('counts') != expected_counts:
        raise ValueError(f'Manifest counts do not match {expected_counts}.')

    table_entries = manifest.get('tables', [])
    if not table_entries:
        raise ValueError('Manifest does not record a prepared CountQA table.')
    table_image_paths = None
    for entry in table_entries:
        path = lmu_data / entry['path']
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            raise ValueError(f'Prepared CountQA table is missing or has wrong size: {path}.')
        if sha256_file(path) != entry['sha256']:
            raise ValueError(f'Prepared CountQA table checksum mismatch: {path}.')
        data = read_table(path)
        if len(data) != COUNTQA_QA_ROWS:
            raise ValueError(f'Prepared CountQA table has {len(data)} rows: {path}.')
        if data['index'].duplicated().any():
            raise ValueError(f'Prepared CountQA table contains duplicate indices: {path}.')
        expected_indices = [
            make_countqa_index(image_row, question_ordinal)
            for image_row, question_ordinal in zip(
                data['source_image_row'], data['question_ordinal']
            )
        ]
        if data['index'].tolist() != expected_indices:
            raise ValueError(f'Prepared CountQA table has non-canonical indices: {path}.')
        if data['source_image_row'].nunique() != COUNTQA_IMAGE_ROWS:
            raise ValueError(f'Prepared CountQA table has the wrong source image count: {path}.')
        current_image_paths = {
            os.fspath(Path('images') / COUNTQA_NAME / image_path)
            for image_path in data['image_path'].unique()
        }
        if table_image_paths is not None and current_image_paths != table_image_paths:
            raise ValueError('Prepared CountQA TSV/CSV files disagree on image paths.')
        table_image_paths = current_image_paths

    image_entries = manifest.get('images', [])
    if len(image_entries) != COUNTQA_IMAGE_ROWS:
        raise ValueError(
            f'Manifest records {len(image_entries)} images, expected {COUNTQA_IMAGE_ROWS}.'
        )
    manifest_image_paths = {entry['path'] for entry in image_entries}
    if manifest_image_paths != table_image_paths:
        raise ValueError(
            'Manifest image inventory does not match the prepared table image_path values.'
        )
    for entry in image_entries:
        path = lmu_data / entry['path']
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            raise ValueError(f'Prepared CountQA image is missing or has wrong size: {path}.')
        if sha256_file(path) != entry['sha256']:
            raise ValueError(f'Prepared CountQA image checksum mismatch: {path}.')

    result = {
        'status': 'verified',
        'revision': COUNTQA_REVISION,
        'image_rows': COUNTQA_IMAGE_ROWS,
        'qa_rows': COUNTQA_QA_ROWS,
        'tables': [entry['path'] for entry in manifest['tables']],
        'manifest': os.fspath(manifest_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def prepare_countqa(lmu_data, output_format='tsv', cache_dir=None):
    try:
        import pyarrow
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            'CountQA preparation requires pyarrow in the Qwen environment.'
        ) from exc
    try:
        import huggingface_hub
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            'CountQA preparation requires huggingface_hub in the Qwen environment.'
        ) from exc

    lmu_data = Path(lmu_data).expanduser().resolve()
    lmu_data.mkdir(parents=True, exist_ok=True)
    images_dir = lmu_data / 'images' / COUNTQA_NAME
    images_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.dataset_info(
        COUNTQA_DATASET_ID,
        revision=COUNTQA_REVISION,
        files_metadata=True,
    )
    if info.sha != COUNTQA_REVISION:
        raise ValueError(f'Hugging Face resolved {info.sha}, expected {COUNTQA_REVISION}.')
    source_entries = []
    for sibling in info.siblings:
        is_split_shard = sibling.rfilename.startswith(f'data/{COUNTQA_SPLIT}-')
        if is_split_shard and sibling.rfilename.endswith('.parquet'):
            source_entries.append(sibling)
    source_entries.sort(key=lambda entry: entry.rfilename)
    if not source_entries:
        raise ValueError('Pinned CountQA revision contains no test Parquet shards.')

    records = []
    image_manifest = []
    source_manifest = []
    image_row = 0
    for source_entry in source_entries:
        parquet_path = Path(hf_hub_download(
            repo_id=COUNTQA_DATASET_ID,
            filename=source_entry.rfilename,
            repo_type='dataset',
            revision=COUNTQA_REVISION,
            cache_dir=cache_dir,
        ))
        parquet_checksum = sha256_file(parquet_path)
        expected_checksum = source_entry.lfs.sha256 if source_entry.lfs is not None else None
        if expected_checksum and parquet_checksum != expected_checksum:
            raise ValueError(f'Source shard checksum mismatch: {source_entry.rfilename}.')
        source_manifest.append({
            'path': source_entry.rfilename,
            'bytes': parquet_path.stat().st_size,
            'sha256': parquet_checksum,
        })

        parquet_file = parquet.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches():
            for source_row in batch.to_pylist():
                content, extension = image_bytes_and_extension(
                    source_row.pop('image'), image_row
                )
                image_name = f'countqa-{image_row:04d}.{extension}'
                image_path = images_dir / image_name
                image_checksum = hashlib.sha256(content).hexdigest()
                if not image_path.is_file() or sha256_file(image_path) != image_checksum:
                    atomic_write_bytes(image_path, content)
                relative_image_path = image_path.relative_to(lmu_data)
                image_manifest.append({
                    'image_row': image_row,
                    'path': os.fspath(relative_image_path),
                    'bytes': len(content),
                    'sha256': image_checksum,
                })
                records.extend(expand_countqa_row(source_row, image_row, image_name))
                image_row += 1

    if image_row != COUNTQA_IMAGE_ROWS:
        raise ValueError(
            f'Pinned CountQA revision yielded {image_row} images, '
            f'expected {COUNTQA_IMAGE_ROWS}.'
        )
    if len(records) != COUNTQA_QA_ROWS:
        raise ValueError(
            f'Pinned CountQA revision yielded {len(records)} QA rows, '
            f'expected {COUNTQA_QA_ROWS}.'
        )

    data = pd.DataFrame(records)
    if data['index'].duplicated().any():
        raise ValueError('Expanded CountQA data contains duplicate canonical indices.')
    serialized = serialize_metadata(data)
    suffixes = ['tsv', 'csv'] if output_format == 'both' else [output_format]
    table_manifest = []
    for suffix in suffixes:
        table_path = lmu_data / f'{COUNTQA_NAME}.{suffix}'
        write_table(serialized, table_path)
        table_manifest.append({
            'path': table_path.name,
            'bytes': table_path.stat().st_size,
            'sha256': sha256_file(table_path),
        })

    manifest = {
        'schema_version': 1,
        'dataset_id': COUNTQA_DATASET_ID,
        'revision': COUNTQA_REVISION,
        'split': COUNTQA_SPLIT,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'counts': {'image_rows': image_row, 'qa_rows': len(records)},
        'tables': table_manifest,
        'images': image_manifest,
        'source_files': source_manifest,
        'preparation_runtime': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'pyarrow': pyarrow.__version__,
            'huggingface_hub': huggingface_hub.__version__,
            'pillow': PIL.__version__,
        },
    }
    manifest_path = lmu_data / f'{COUNTQA_NAME}.manifest.json'
    atomic_write_json(manifest_path, manifest)
    return verify_prepared_dataset(lmu_data, manifest_path)


def parse_args():
    default_lmu_data = os.environ.get('LMUData', os.fspath(Path.home() / 'LMUData'))
    parser = argparse.ArgumentParser(
        description='Prepare pinned CountQA Parquet shards for offline VLMEvalKit inference.'
    )
    parser.add_argument('--lmu-data', default=default_lmu_data, help='LMUData output root.')
    parser.add_argument(
        '--format',
        choices=('tsv', 'csv', 'both'),
        default='tsv',
        help='Prepared annotation table format.',
    )
    parser.add_argument('--cache-dir', default=None, help='Optional Hugging Face cache directory.')
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Verify an existing manifest and all prepared file checksums without network access.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verify_only:
        verify_prepared_dataset(args.lmu_data)
    else:
        prepare_countqa(args.lmu_data, output_format=args.format, cache_dir=args.cache_dir)


if __name__ == '__main__':
    main()
