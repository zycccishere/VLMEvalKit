#!/usr/bin/env python3
import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image


REPO_ID = 'Jayant-Sravan/CountQA'
REVISION = 'f92cc6fe46542c61e2916e3d2ae9a911e2216b1a'
EXPECTED_IMAGES = 1001
EXPECTED_ROWS = 1528


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_index(image_row, ordinal):
    return f'countqa-{image_row:04d}-{ordinal:02d}'


def image_bytes_and_extension(value, image_row):
    content = value.get('bytes') if isinstance(value, dict) else None
    if not isinstance(content, (bytes, bytearray)):
        raise ValueError(f'CountQA image row {image_row} has no embedded image bytes.')
    content = bytes(content)
    with Image.open(io.BytesIO(content)) as image:
        image_format = (image.format or '').upper()
        image.verify()
    extension = {'JPEG': 'jpg', 'PNG': 'png', 'WEBP': 'webp'}.get(image_format)
    if extension is None:
        raise ValueError(f'Unsupported CountQA image format: {image_format}')
    return content, extension


def expand_row(row, image_row, image_name):
    questions, answers = row.get('questions'), row.get('answers')
    if not isinstance(questions, list) or not isinstance(answers, list):
        raise TypeError(f'CountQA image row {image_row} questions/answers must be lists.')
    if len(questions) != len(answers):
        raise ValueError(f'CountQA image row {image_row} has mismatched questions and answers.')
    records = []
    for ordinal, (question, answer) in enumerate(zip(questions, answers)):
        answer = str(answer).strip()
        if not answer.isascii() or not answer.isdigit():
            raise ValueError(f'CountQA answer is not a non-negative integer: {answer!r}')
        records.append({
            'index': make_index(image_row, ordinal),
            'source_image_row': image_row,
            'question_ordinal': ordinal,
            'image_path': image_name,
            'question': str(question).strip(),
            'answer': str(int(answer)),
        })
    return records


def validate(lmu_data, expected_images=EXPECTED_IMAGES, expected_rows=EXPECTED_ROWS):
    lmu_data = Path(lmu_data)
    path = lmu_data / 'CountQA.tsv'
    data = pd.read_csv(path, sep='\t', dtype={'index': str, 'answer': str})
    required = {'index', 'source_image_row', 'question_ordinal', 'image_path', 'question', 'answer'}
    missing = sorted(required - set(data.columns))
    if missing or len(data) != expected_rows or data['index'].duplicated().any():
        raise ValueError(f'Invalid CountQA TSV: missing={missing}, rows={len(data)}')
    if data['source_image_row'].nunique() != expected_images:
        raise ValueError('CountQA TSV has the wrong source-image count.')
    expected = [make_index(row, ordinal) for row, ordinal in zip(
        data['source_image_row'], data['question_ordinal'])]
    if data['index'].tolist() != expected:
        raise ValueError('CountQA TSV contains non-canonical indices.')
    image_root = lmu_data / 'images' / 'CountQA'
    missing_images = [name for name in data['image_path'].unique() if not (image_root / name).is_file()]
    if missing_images:
        raise FileNotFoundError(f'CountQA image is missing: {missing_images[0]}')
    if any(not str(answer).isascii() or not str(answer).isdigit() for answer in data['answer']):
        raise ValueError('CountQA TSV contains a non-integer answer.')
    return data


def prepare(lmu_data, cache_dir=None):
    import pyarrow.parquet as parquet
    from huggingface_hub import HfApi, hf_hub_download

    lmu_data = Path(lmu_data).expanduser().resolve()
    image_root = lmu_data / 'images' / 'CountQA'
    image_root.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(REPO_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError(f'Hugging Face resolved {info.sha}, expected {REVISION}.')
    shards = sorted(
        (
            sibling for sibling in info.siblings
            if sibling.rfilename.startswith('data/test-')
            and sibling.rfilename.endswith('.parquet')
        ),
        key=lambda sibling: sibling.rfilename,
    )
    records, image_row = [], 0
    for shard in shards:
        path = Path(hf_hub_download(
            repo_id=REPO_ID,
            filename=shard.rfilename,
            repo_type='dataset',
            revision=REVISION,
            cache_dir=cache_dir,
        ))
        if shard.lfs is not None and sha256(path) != shard.lfs.sha256:
            raise ValueError(f'CountQA source checksum mismatch: {shard.rfilename}')
        for batch in parquet.ParquetFile(path).iter_batches():
            for row in batch.to_pylist():
                content, extension = image_bytes_and_extension(row.pop('image'), image_row)
                image_name = f'countqa-{image_row:04d}.{extension}'
                image_path = image_root / image_name
                if not image_path.is_file() or image_path.read_bytes() != content:
                    image_path.write_bytes(content)
                records.extend(expand_row(row, image_row, image_name))
                image_row += 1
    if image_row != EXPECTED_IMAGES or len(records) != EXPECTED_ROWS:
        raise ValueError(f'Pinned CountQA yielded {image_row} images and {len(records)} rows.')
    target = lmu_data / 'CountQA.tsv'
    temporary = target.with_name(f'.{target.name}.tmp')
    pd.DataFrame(records).to_csv(temporary, sep='\t', index=False)
    os.replace(temporary, target)
    manifest = {
        'dataset_id': REPO_ID,
        'revision': REVISION,
        'counts': {'image_rows': image_row, 'qa_rows': len(records)},
        'table_sha256': sha256(target),
    }
    (lmu_data / 'CountQA.manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return validate(lmu_data)


def main():
    parser = argparse.ArgumentParser(description='Prepare CountQA for VLMEvalKit.')
    parser.add_argument('--lmu-data', default=os.environ.get('LMUData', str(Path.home() / 'LMUData')))
    parser.add_argument('--cache-dir')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    data = validate(args.lmu_data) if args.verify_only else prepare(args.lmu_data, args.cache_dir)
    print(json.dumps({'rows': len(data), 'images': data['source_image_row'].nunique()}, sort_keys=True))


if __name__ == '__main__':
    main()
