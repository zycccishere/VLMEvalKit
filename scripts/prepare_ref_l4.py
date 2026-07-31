#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import os.path as osp
import shutil
import sys
import tarfile
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault('VLMEVAL_LAZY_INIT', '1')
os.environ.setdefault('VLMEVAL_VLM_MINIMAL_IMPORT', '1')
os.environ.setdefault('VLMEVAL_API_MINIMAL_IMPORT', '1')

import vlmeval  # noqa: E402,F401


# Preparation should not import optional dependencies for unrelated datasets.
dataset_package = types.ModuleType('vlmeval.dataset')
dataset_package.__path__ = [str(REPO_ROOT / 'vlmeval' / 'dataset')]
dataset_package.__package__ = 'vlmeval.dataset'
sys.modules['vlmeval.dataset'] = dataset_package

from vlmeval.dataset.ref_l4 import (  # noqa: E402
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_TOTAL_ROWS,
    HF_DATASET_ID,
    HF_REVISION,
    PREPARED_MANIFEST,
    PREPARED_SCHEMA_VERSION,
    PREPARED_TSV,
    RefL4PreparationError,
    UPSTREAM_CODE_REVISION,
    _read_prepared_package,
    prepared_root,
)
from vlmeval.smp.file import LMUDataRoot  # noqa: E402


SOURCE_FILES = {
    'val': {
        'filename': 'ref-l4-val.parquet',
        'sha256': '7aef079b3fa1ec5de7548774ae6206f375f4b189fd5acecae86741a452da1926',
    },
    'test': {
        'filename': 'ref-l4-test.parquet',
        'sha256': '6af2bcac60d0aff5f78c0a0c834377e43ceb17d1e11cbaacdf4eda5f9a9e0068',
    },
}
IMAGE_ARCHIVE = 'images.tar.gz'
IMAGE_ARCHIVE_SIZE = 3512747994
IMAGE_ARCHIVE_SHA256 = 'a07c7b85e94d3dcd1f7847ba85d670590b86d6000ffa3c8dd11e988b2af2a7b7'
SOURCE_COLUMNS = {
    'id', 'caption', 'bbox', 'bbox_area', 'bbox_id', 'ori_category_id',
    'image_id', 'height', 'width', 'file_name', 'is_rewrite', 'split',
}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_source_frame(data: pd.DataFrame, split: str) -> None:
    missing = sorted(SOURCE_COLUMNS - set(data.columns))
    if missing:
        raise RefL4PreparationError(f'{split} parquet is missing source columns: {missing}')
    if len(data) != EXPECTED_SPLIT_COUNTS[split]:
        raise RefL4PreparationError(
            f'{split} parquet has {len(data)} rows; expected {EXPECTED_SPLIT_COUNTS[split]}.'
        )
    if set(data['split'].astype(str)) != {split}:
        raise RefL4PreparationError(f'{split} parquet contains mismatched split labels.')
    if data['id'].duplicated().any():
        raise RefL4PreparationError(f'{split} parquet contains duplicate annotation ids.')

    for row in data.to_dict('records'):
        bbox = row['bbox']
        if isinstance(bbox, (str, bytes)) or not hasattr(bbox, '__len__') or len(bbox) != 4:
            raise RefL4PreparationError(f"Annotation {row['id']} has an invalid xywh bbox.")
        x, y, width, height = (float(value) for value in bbox)
        values = (x, y, width, height, float(row['width']), float(row['height']))
        if not all(pd.notna(value) and abs(value) != float('inf') for value in values):
            raise RefL4PreparationError(f"Annotation {row['id']} has non-finite geometry.")
        if width <= 0 or height <= 0 or row['width'] <= 0 or row['height'] <= 0:
            raise RefL4PreparationError(f"Annotation {row['id']} has degenerate geometry.")
        if abs(width * height - float(row['bbox_area'])) > max(1e-6, abs(row['bbox_area']) * 1e-9):
            raise RefL4PreparationError(f"Annotation {row['id']} has inconsistent bbox_area.")


def build_prepared_frame(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    ordered = []
    for split in ('val', 'test'):
        source = frames[split]
        validate_source_frame(source, split)
        bbox = pd.DataFrame(source['bbox'].tolist(), columns=['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'])
        prepared = pd.DataFrame({
            'index': source['id'].astype(str),
            'annotation_id': source['id'].astype('int64'),
            'question': source['caption'].astype(str),
            'image_path': [osp.join('images', str(name)) for name in source['file_name']],
            'bbox_x': bbox['bbox_x'].astype(float),
            'bbox_y': bbox['bbox_y'].astype(float),
            'bbox_w': bbox['bbox_w'].astype(float),
            'bbox_h': bbox['bbox_h'].astype(float),
            'bbox_area': source['bbox_area'].astype(float),
            'bbox_id': source['bbox_id'].astype(str),
            'ori_category_id': source['ori_category_id'].astype(str),
            'image_id': source['image_id'].astype(str),
            'height': source['height'].astype('int64'),
            'width': source['width'].astype('int64'),
            'is_rewrite': source['is_rewrite'].astype(bool),
            'split': source['split'].astype(str),
            'source_revision': HF_REVISION,
        })
        ordered.append(prepared)

    result = pd.concat(ordered, ignore_index=True)
    if len(result) != EXPECTED_TOTAL_ROWS:
        raise RefL4PreparationError(f'Combined source has {len(result)} rows; expected {EXPECTED_TOTAL_ROWS}.')
    if result['annotation_id'].duplicated().any():
        raise RefL4PreparationError('Combined source contains duplicate annotation ids.')
    expected_ids = set(range(EXPECTED_TOTAL_ROWS))
    actual_ids = set(int(value) for value in result['annotation_id'])
    if actual_ids != expected_ids:
        raise RefL4PreparationError('Combined annotation ids are not the canonical contiguous range 0..45340.')
    return result


def expected_images(data: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
    expected: Dict[str, Tuple[int, int]] = {}
    for row in data.to_dict('records'):
        relative_path = str(row['image_path'])
        if not relative_path.startswith(f'images{os.sep}'):
            raise RefL4PreparationError(f'Unsafe prepared image path: {relative_path}')
        member_name = relative_path[len(f'images{os.sep}'):]
        dimensions = (int(row['width']), int(row['height']))
        previous = expected.setdefault(member_name, dimensions)
        if previous != dimensions:
            raise RefL4PreparationError(
                f'Image {member_name} has conflicting annotation dimensions: {previous} vs {dimensions}.'
            )
    return expected


def extract_and_verify_images(archive_path: str, image_root: str, expected: Dict[str, Tuple[int, int]]) -> list:
    os.makedirs(image_root, exist_ok=True)
    pending = set(expected)
    seen = set()
    inventory = []
    real_image_root = osp.realpath(image_root)

    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected:
                continue
            if member.name in seen:
                raise RefL4PreparationError(f'Image archive contains duplicate member: {member.name}')

            destination = osp.realpath(osp.join(image_root, member.name))
            if osp.commonpath([real_image_root, destination]) != real_image_root:
                raise RefL4PreparationError(f'Image archive contains an unsafe path: {member.name}')
            os.makedirs(osp.dirname(destination), exist_ok=True)

            source = archive.extractfile(member)
            if source is None:
                raise RefL4PreparationError(f'Could not read image archive member: {member.name}')
            temporary = destination + '.part'
            with source, open(temporary, 'wb') as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            os.replace(temporary, destination)

            with Image.open(destination) as image:
                actual_size = image.size
                image.verify()
            if actual_size != expected[member.name]:
                raise RefL4PreparationError(
                    f'Image {member.name} has size {actual_size}; expected {expected[member.name]}.'
                )
            inventory.append({
                'path': osp.join('images', member.name),
                'bytes': osp.getsize(destination),
                'sha256': sha256(destination),
            })
            seen.add(member.name)
            pending.remove(member.name)
            if not pending:
                break

    if pending:
        sample = ', '.join(sorted(pending)[:5])
        raise RefL4PreparationError(f'Image archive is missing {len(pending)} referenced files: {sample}')
    return sorted(inventory, key=lambda entry: entry['path'])


def atomic_json_dump(payload: dict, path: str) -> None:
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def install_stage(stage_root: str, final_root: str, force: bool) -> None:
    if not osp.exists(final_root):
        os.replace(stage_root, final_root)
        return
    if not force:
        raise FileExistsError(f'Prepared Ref-L4 package already exists at {final_root}.')

    backup = final_root + f'.backup-{os.getpid()}'
    os.replace(final_root, backup)
    try:
        os.replace(stage_root, final_root)
    except Exception:
        os.replace(backup, final_root)
        raise
    shutil.rmtree(backup)


def prepare(data_root: str, force: bool = False) -> dict:
    final_root = prepared_root(data_root)
    if osp.isdir(final_root) and not force:
        _, manifest = _read_prepared_package(final_root, check_images=True)
        return manifest

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError('Preparation requires `huggingface_hub`.') from exc

    source_paths = {}
    for split, metadata in SOURCE_FILES.items():
        path = hf_hub_download(
            repo_id=HF_DATASET_ID,
            filename=metadata['filename'],
            repo_type='dataset',
            revision=HF_REVISION,
        )
        actual_sha = sha256(path)
        if actual_sha != metadata['sha256']:
            raise RefL4PreparationError(
                f"Pinned {metadata['filename']} checksum is {actual_sha}; expected {metadata['sha256']}."
            )
        source_paths[split] = path

    archive_path = hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename=IMAGE_ARCHIVE,
        repo_type='dataset',
        revision=HF_REVISION,
    )
    if osp.getsize(archive_path) != IMAGE_ARCHIVE_SIZE:
        raise RefL4PreparationError(
            f'Pinned image archive has {osp.getsize(archive_path)} bytes; expected {IMAGE_ARCHIVE_SIZE}.'
        )
    archive_sha256 = sha256(archive_path)
    if archive_sha256 != IMAGE_ARCHIVE_SHA256:
        raise RefL4PreparationError(
            f'Pinned image archive checksum is {archive_sha256}; expected {IMAGE_ARCHIVE_SHA256}.'
        )

    frames = {split: pd.read_parquet(path) for split, path in source_paths.items()}
    data = build_prepared_frame(frames)
    image_map = expected_images(data)

    dataset_root = osp.dirname(final_root)
    os.makedirs(dataset_root, exist_ok=True)
    stage_root = tempfile.mkdtemp(prefix=f'.{HF_REVISION}.prepare-', dir=dataset_root)
    try:
        image_inventory = extract_and_verify_images(
            archive_path, osp.join(stage_root, 'images'), image_map
        )

        tsv_path = osp.join(stage_root, PREPARED_TSV)
        temporary_tsv = tsv_path + '.tmp'
        data.to_csv(
            temporary_tsv,
            sep='\t',
            index=False,
            encoding='utf-8',
            quoting=csv.QUOTE_ALL,
        )
        os.replace(temporary_tsv, tsv_path)

        manifest = {
            'schema_version': PREPARED_SCHEMA_VERSION,
            'source_repo': HF_DATASET_ID,
            'source_revision': HF_REVISION,
            'upstream_code_revision': UPSTREAM_CODE_REVISION,
            'source_files': {
                split: {
                    'filename': SOURCE_FILES[split]['filename'],
                    'sha256': SOURCE_FILES[split]['sha256'],
                    'row_count': EXPECTED_SPLIT_COUNTS[split],
                }
                for split in ('val', 'test')
            },
            'image_archive': {
                'filename': IMAGE_ARCHIVE,
                'size_bytes': IMAGE_ARCHIVE_SIZE,
                'sha256': IMAGE_ARCHIVE_SHA256,
            },
            'image_files': image_inventory,
            'annotation_file': PREPARED_TSV,
            'annotation_sha256': sha256(tsv_path),
            'row_count': len(data),
            'split_counts': EXPECTED_SPLIT_COUNTS,
            'image_file_count': len(image_map),
            'file_validation': {
                'expected': len(image_map),
                'checked': len(image_map),
                'missing': 0,
                'dimension_mismatches': 0,
            },
            'bbox_source_format': 'xywh_original_pixel',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
        }
        atomic_json_dump(manifest, osp.join(stage_root, PREPARED_MANIFEST))

        _read_prepared_package(stage_root, check_images=True)
        install_stage(stage_root, final_root, force=force)
        stage_root = ''
        return manifest
    finally:
        if stage_root and osp.isdir(stage_root):
            shutil.rmtree(stage_root)


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare pinned Ref-L4 assets for VLMEvalKit.')
    parser.add_argument('--data-root', default=LMUDataRoot(), help='LMUData root directory.')
    parser.add_argument('--force', action='store_true', help='Atomically replace an existing prepared package.')
    args = parser.parse_args()
    manifest = prepare(osp.abspath(args.data_root), force=args.force)
    print(json.dumps({'prepared_root': prepared_root(args.data_root), 'manifest': manifest}, indent=2))


if __name__ == '__main__':
    main()
