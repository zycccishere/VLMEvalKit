import hashlib
import json
import os
import os.path as osp
import re
import shutil
import string
import tempfile
import uuid
from datetime import datetime, timezone

import pandas as pd

from .image_base import ImageBaseDataset
from ..smp.file import LMUDataRoot, dump, get_intermediate_file_path, load
from ..smp.misc import toliststr


class SpatialMQA(ImageBaseDataset):
    """SpatialMQA with a pinned, locally prepared test split."""

    TYPE = 'MCQ'
    FORCE_DATASET_PROMPT = True

    HF_REPO_ID = 'liuziyan/SpatialMQA'
    HF_REVISION = '2c297135743209b91fe0729c033b49bb1d72f788'
    HF_CONFIG = 'default'
    HF_SPLIT = 'test'
    HF_SPLIT_FILE = 'test.jsonl'
    HF_SPLIT_SHA256 = 'f5e3a76059087dba72b9f9396981f0c3df128b6335f9aa12655ea580b691ba9e'
    HF_SPLIT_SIZES = {'train': 3780, 'validation': 536, 'test': 1076}
    HF_SCHEMA_COLUMNS = ['image', 'question', 'options', 'answer', ',answer']

    PREPARED_DIRNAME = 'SpatialMQA'
    PREPARED_MANIFEST = 'manifest.json'
    PREPARED_DATA_FILE = 'SpatialMQA.tsv'
    PREPARED_FORMAT_VERSION = 1

    RELATION_TO_CATEGORY = {
        'on/above': 'y',
        'below': 'y',
        'in front of': 'z',
        'behind': 'z',
        'left of': 'x',
        'right of': 'x',
    }
    RELATION_ORDER = list(RELATION_TO_CATEGORY)

    @classmethod
    def supported_datasets(cls):
        return ['SpatialMQA']

    @staticmethod
    def _normalize_text(value):
        return ' '.join(str(value).strip().split()).casefold()

    @staticmethod
    def _is_missing(value):
        return value is None or (isinstance(value, float) and pd.isna(value))

    @classmethod
    def _coerce_options(cls, value):
        if isinstance(value, (list, tuple)):
            options = list(value)
        elif isinstance(value, str):
            try:
                options = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError('SpatialMQA options must be a JSON list') from exc
        else:
            raise ValueError(f'SpatialMQA options must be a list, got {type(value).__name__}')

        if not 2 <= len(options) <= 6:
            raise ValueError(f'SpatialMQA options must contain 2-6 items, got {len(options)}')
        if any(not isinstance(option, str) or not option.strip() for option in options):
            raise ValueError('SpatialMQA options must be non-empty strings')

        normalized = [cls._normalize_text(option) for option in options]
        if len(normalized) != len(set(normalized)):
            raise ValueError('SpatialMQA options must be unique after normalization')
        unknown = [option for option in normalized if option not in cls.RELATION_TO_CATEGORY]
        if unknown:
            raise ValueError(f'Unknown SpatialMQA relation choices: {unknown}')
        return [option.strip() for option in options]

    @classmethod
    def _canonicalize_answer(cls, options, answer):
        options = cls._coerce_options(options)
        if cls._is_missing(answer) or not str(answer).strip():
            raise ValueError('SpatialMQA gold answer is missing')

        answer_text = str(answer).strip()
        candidates = set()
        letter_match = re.fullmatch(r'[\(\[]?\s*([A-Fa-f])\s*[\)\]]?', answer_text)
        if letter_match is not None:
            letter_index = ord(letter_match.group(1).upper()) - ord('A')
            if letter_index < len(options):
                candidates.add(letter_index)

        normalized_answer = cls._normalize_text(answer_text)
        for option_index, option in enumerate(options):
            if cls._normalize_text(option) == normalized_answer:
                candidates.add(option_index)

        if len(candidates) != 1:
            raise ValueError(
                'SpatialMQA gold must resolve to exactly one current choice; '
                f'answer={answer!r}, candidates={sorted(candidates)}'
            )

        option_index = candidates.pop()
        return string.ascii_uppercase[option_index], option_index, options[option_index]

    @classmethod
    def _records_to_dataframe(cls, records, image_root, split='test'):
        rows = []
        image_root = osp.abspath(image_root)
        for source_row, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f'SpatialMQA row {source_row} is not an object')
            missing = [key for key in ('image', 'question', 'options', 'answer') if key not in record]
            if missing:
                raise ValueError(f'SpatialMQA row {source_row} is missing fields: {missing}')

            image_name = str(record['image']).strip()
            if not re.fullmatch(r'[0-9]{12}\.jpg', image_name):
                raise ValueError(f'Invalid SpatialMQA image name at row {source_row}: {image_name!r}')
            image_path = osp.join(image_root, image_name)
            if not osp.isfile(image_path):
                raise FileNotFoundError(f'Missing SpatialMQA image at row {source_row}: {image_path}')

            question = str(record['question']).strip()
            if not question:
                raise ValueError(f'SpatialMQA question is empty at row {source_row}')

            options = cls._coerce_options(record['options'])
            answer, _, relation = cls._canonicalize_answer(options, record['answer'])
            anomalous_answer = record.get(',answer')
            if not cls._is_missing(anomalous_answer) and str(anomalous_answer).strip():
                anomaly_letter, _, _ = cls._canonicalize_answer(options, anomalous_answer)
                if anomaly_letter != answer:
                    raise ValueError(
                        f'Conflicting answer and ,answer values at SpatialMQA row {source_row}'
                    )

            row = {
                'index': f'spatialmqa-{split}-{source_row:06d}',
                'image_path': image_path,
                'question': question,
                'options': options,
                'answer': answer,
                'relation': relation,
                'category': cls.RELATION_TO_CATEGORY[cls._normalize_text(relation)],
                'split': split,
                'source_row': source_row,
            }
            for option_index, option in enumerate(options):
                row[string.ascii_uppercase[option_index]] = option
            rows.append(row)

        data = pd.DataFrame(rows)
        if data.empty:
            raise ValueError('SpatialMQA source contains no rows')
        if data['index'].duplicated().any():
            raise ValueError('SpatialMQA stable indices are not unique')
        return data

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _resolve_prepared_path(cls, prepared_root, relative_path):
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError('SpatialMQA manifest path is missing')
        if osp.isabs(relative_path):
            raise ValueError('SpatialMQA manifest paths must be relative to the prepared root')
        prepared_root = osp.realpath(prepared_root)
        resolved = osp.realpath(osp.join(prepared_root, relative_path))
        if osp.commonpath([prepared_root, resolved]) != prepared_root:
            raise ValueError(f'SpatialMQA manifest path escapes prepared root: {relative_path}')
        return resolved

    @classmethod
    def _validate_prepared_manifest(cls, prepared_root, manifest, expected_rows=None):
        if manifest.get('format_version') != cls.PREPARED_FORMAT_VERSION:
            raise ValueError('Unsupported SpatialMQA prepared artifact format')

        source = manifest.get('source', {})
        expected_source = {
            'repo_id': cls.HF_REPO_ID,
            'revision': cls.HF_REVISION,
            'config': cls.HF_CONFIG,
            'split': cls.HF_SPLIT,
            'file': cls.HF_SPLIT_FILE,
            'sha256': cls.HF_SPLIT_SHA256,
        }
        for key, expected in expected_source.items():
            if source.get(key) != expected:
                raise ValueError(
                    f'SpatialMQA manifest source {key} mismatch: '
                    f'{source.get(key)!r} != {expected!r}'
                )

        row_count = manifest.get('row_count')
        if expected_rows is not None and row_count != expected_rows:
            raise ValueError(f'SpatialMQA manifest row count is {row_count}, expected {expected_rows}')

        checks = manifest.get('checks', {})
        required_checks = {
            'data_file_exists': True,
            'source_file_exists': True,
            'all_image_paths_exist': True,
            'duplicate_index_count': 0,
            'missing_image_count': 0,
        }
        for key, expected in required_checks.items():
            if checks.get(key) != expected:
                raise ValueError(f'SpatialMQA manifest check {key} is not {expected!r}')

        data_path = cls._resolve_prepared_path(prepared_root, manifest.get('data_file'))
        source_path = cls._resolve_prepared_path(prepared_root, manifest.get('source_file'))
        image_root = cls._resolve_prepared_path(prepared_root, manifest.get('image_root'))
        for label, path in [('data file', data_path), ('source file', source_path)]:
            if not osp.isfile(path):
                raise FileNotFoundError(f'SpatialMQA prepared {label} is missing: {path}')
        if not osp.isdir(image_root):
            raise FileNotFoundError(f'SpatialMQA prepared image directory is missing: {image_root}')
        if cls._sha256(data_path) != manifest.get('data_sha256'):
            raise ValueError('SpatialMQA prepared TSV checksum mismatch')
        if cls._sha256(source_path) != manifest.get('source_sha256'):
            raise ValueError('SpatialMQA canonical source checksum mismatch')

        data = pd.read_csv(data_path, sep='\t', dtype={'index': str})
        required_columns = {
            'index', 'image_path', 'question', 'options', 'answer',
            'relation', 'category', 'split', 'source_row',
        }
        missing_columns = sorted(required_columns - set(data.columns))
        if missing_columns:
            raise ValueError(f'SpatialMQA prepared TSV is missing columns: {missing_columns}')
        if len(data) != row_count:
            raise ValueError(f'SpatialMQA prepared TSV has {len(data)} rows, manifest says {row_count}')
        if data['index'].duplicated().any():
            duplicates = data.loc[data['index'].duplicated(keep=False), 'index'].tolist()
            raise ValueError(f'SpatialMQA prepared TSV has duplicate indices: {duplicates[:5]}')
        expected_indices = [f'spatialmqa-{cls.HF_SPLIT}-{row:06d}' for row in range(len(data))]
        if data['index'].tolist() != expected_indices:
            raise ValueError('SpatialMQA prepared stable index sequence is invalid')
        if data['source_row'].tolist() != list(range(len(data))):
            raise ValueError('SpatialMQA prepared source_row sequence is invalid')
        if set(data['split'].astype(str)) != {cls.HF_SPLIT}:
            raise ValueError('SpatialMQA prepared split column is invalid')
        if checks.get('pil_verified_image_count') != manifest.get('unique_image_count'):
            raise ValueError('SpatialMQA PIL-verified image count mismatch')

        image_entries = manifest.get('image_files', [])
        if len(image_entries) != manifest.get('unique_image_count'):
            raise ValueError('SpatialMQA manifest image inventory count mismatch')
        inventory_paths = set()
        for entry in image_entries:
            image_path = cls._resolve_prepared_path(prepared_root, entry.get('path'))
            inventory_paths.add(osp.realpath(image_path))
            if not osp.isfile(image_path):
                raise FileNotFoundError(f'SpatialMQA prepared image is missing: {image_path}')
            if osp.getsize(image_path) != entry.get('bytes'):
                raise ValueError(f'SpatialMQA prepared image size mismatch: {image_path}')
            if cls._sha256(image_path) != entry.get('sha256'):
                raise ValueError(f'SpatialMQA prepared image checksum mismatch: {image_path}')

        parsed_options = []
        missing_images = []
        image_root_real = osp.realpath(image_root)
        for row_number, row in data.iterrows():
            options = cls._coerce_options(row['options'])
            gold, _, relation = cls._canonicalize_answer(options, row['answer'])
            if gold != str(row['answer']).strip().upper():
                raise ValueError(f'Non-canonical SpatialMQA gold at prepared row {row_number}')
            if cls._normalize_text(relation) != cls._normalize_text(row['relation']):
                raise ValueError(f'SpatialMQA relation mismatch at prepared row {row_number}')
            expected_category = cls.RELATION_TO_CATEGORY[cls._normalize_text(relation)]
            if str(row['category']).strip() != expected_category:
                raise ValueError(f'SpatialMQA category mismatch at prepared row {row_number}')

            image_path = osp.realpath(str(row['image_path']))
            if osp.commonpath([image_root_real, image_path]) != image_root_real:
                raise ValueError(f'SpatialMQA image path escapes image root: {image_path}')
            if not osp.isfile(image_path):
                missing_images.append(image_path)
            parsed_options.append(options)

        if missing_images:
            raise FileNotFoundError(
                f'SpatialMQA prepared TSV references {len(missing_images)} missing images; '
                f'first={missing_images[0]}'
            )
        if data['image_path'].nunique() != manifest.get('unique_image_count'):
            raise ValueError('SpatialMQA prepared unique image count mismatch')
        if set(osp.realpath(path) for path in data['image_path'].unique()) != inventory_paths:
            raise ValueError('SpatialMQA table images do not match the manifest inventory')

        data['options'] = parsed_options
        return data

    @classmethod
    def validate_prepared(cls, prepared_root=None, expected_rows=None):
        if prepared_root is None:
            prepared_root = osp.join(LMUDataRoot(), cls.PREPARED_DIRNAME)
        manifest_path = osp.join(prepared_root, cls.PREPARED_MANIFEST)
        if not osp.isfile(manifest_path):
            raise FileNotFoundError(
                f'SpatialMQA is not prepared at {prepared_root}. '
                'Run: python scripts/prepare_spatialmqa.py'
            )
        with open(manifest_path, 'r', encoding='utf-8') as stream:
            manifest = json.load(stream)
        data = cls._validate_prepared_manifest(prepared_root, manifest, expected_rows=expected_rows)
        return data, manifest

    @classmethod
    def _atomic_write_json(cls, payload, path):
        os.makedirs(osp.dirname(path), exist_ok=True)
        handle, temporary_path = tempfile.mkstemp(
            prefix=f'.{osp.basename(path)}.', suffix='.tmp', dir=osp.dirname(path)
        )
        try:
            with os.fdopen(handle, 'w', encoding='utf-8') as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        except Exception:
            if osp.exists(temporary_path):
                os.unlink(temporary_path)
            raise

    @classmethod
    def _publish_snapshot(cls, annotation_path, snapshot_root, lmu_data_root, expected_rows):
        annotation_sha256 = cls._sha256(annotation_path)
        if annotation_sha256 != cls.HF_SPLIT_SHA256:
            raise ValueError(
                f'Pinned SpatialMQA source checksum is {annotation_sha256}; '
                f'expected {cls.HF_SPLIT_SHA256}.'
            )
        with open(annotation_path, 'r', encoding='utf-8') as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if len(records) != expected_rows:
            raise ValueError(
                f'Pinned SpatialMQA {cls.HF_SPLIT} has {len(records)} rows, expected {expected_rows}'
            )

        prepared_root = osp.abspath(osp.join(lmu_data_root, cls.PREPARED_DIRNAME))
        revisions_root = osp.join(prepared_root, 'revisions')
        os.makedirs(revisions_root, exist_ok=True)
        staging_dir = tempfile.mkdtemp(prefix='.prepare-', dir=revisions_root)
        artifact_name = f'{cls.HF_REVISION}-{uuid.uuid4().hex[:12]}'
        artifact_dir = osp.join(revisions_root, artifact_name)

        try:
            staging_image_root = osp.join(staging_dir, 'images')
            staging_source_root = osp.join(staging_dir, 'source')
            os.makedirs(staging_image_root)
            os.makedirs(staging_source_root)
            source_copy = osp.join(staging_source_root, cls.HF_SPLIT_FILE)
            shutil.copy2(annotation_path, source_copy)

            image_names = sorted({str(record.get('image', '')).strip() for record in records})
            image_inventory = []
            from PIL import Image

            for image_name in image_names:
                if not re.fullmatch(r'[0-9]{12}\.jpg', image_name):
                    raise ValueError(f'Invalid SpatialMQA image name: {image_name!r}')
                source_image = osp.join(snapshot_root, 'images', image_name)
                if not osp.isfile(source_image):
                    raise FileNotFoundError(f'Pinned SpatialMQA image is missing: {source_image}')
                target_image = osp.join(staging_image_root, image_name)
                shutil.copy2(source_image, target_image)
                with Image.open(target_image) as image:
                    image.verify()
                image_inventory.append({
                    'name': image_name,
                    'bytes': osp.getsize(target_image),
                    'sha256': cls._sha256(target_image),
                })

            data = cls._records_to_dataframe(records, staging_image_root, split=cls.HF_SPLIT)
            final_image_root = osp.join(artifact_dir, 'images')
            data['image_path'] = [
                osp.join(final_image_root, osp.basename(path)) for path in data['image_path']
            ]
            serialized = data.copy()
            serialized['options'] = [
                json.dumps(options, ensure_ascii=True) for options in serialized['options']
            ]
            staging_data_path = osp.join(staging_dir, cls.PREPARED_DATA_FILE)
            serialized.to_csv(staging_data_path, sep='\t', index=False)

            os.replace(staging_dir, artifact_dir)
            artifact_rel = osp.relpath(artifact_dir, prepared_root)
            data_rel = osp.join(artifact_rel, cls.PREPARED_DATA_FILE)
            source_rel = osp.join(artifact_rel, 'source', cls.HF_SPLIT_FILE)
            image_root_rel = osp.join(artifact_rel, 'images')
            final_data_path = osp.join(prepared_root, data_rel)
            final_source_path = osp.join(prepared_root, source_rel)

            manifest = {
                'format_version': cls.PREPARED_FORMAT_VERSION,
                'dataset': 'SpatialMQA',
                'created_at': datetime.now(timezone.utc).isoformat(),
                'source': {
                    'repo_id': cls.HF_REPO_ID,
                    'revision': cls.HF_REVISION,
                    'config': cls.HF_CONFIG,
                    'split': cls.HF_SPLIT,
                    'file': cls.HF_SPLIT_FILE,
                    'sha256': cls.HF_SPLIT_SHA256,
                    'known_split_rows': cls.HF_SPLIT_SIZES,
                    'viewer_schema_columns': cls.HF_SCHEMA_COLUMNS,
                },
                'row_count': len(data),
                'unique_image_count': int(data['image_path'].nunique()),
                'artifact_dir': artifact_rel,
                'data_file': data_rel,
                'source_file': source_rel,
                'image_root': image_root_rel,
                'image_files': [
                    {
                        'path': osp.join(image_root_rel, entry['name']),
                        'bytes': entry['bytes'],
                        'sha256': entry['sha256'],
                    }
                    for entry in image_inventory
                ],
                'data_sha256': cls._sha256(final_data_path),
                'source_sha256': cls._sha256(final_source_path),
                'checks': {
                    'data_file_exists': osp.isfile(final_data_path),
                    'source_file_exists': osp.isfile(final_source_path),
                    'all_image_paths_exist': bool(data['image_path'].map(osp.isfile).all()),
                    'duplicate_index_count': int(data['index'].duplicated().sum()),
                    'missing_image_count': int((~data['image_path'].map(osp.isfile)).sum()),
                    'pil_verified_image_count': len(image_names),
                },
            }

            cls._validate_prepared_manifest(
                prepared_root, manifest, expected_rows=expected_rows
            )
            cls._atomic_write_json(manifest, osp.join(prepared_root, cls.PREPARED_MANIFEST))
            return manifest
        except Exception:
            if osp.isdir(staging_dir):
                shutil.rmtree(staging_dir)
            raise

    @classmethod
    def prepare(cls, lmu_data_root=None, force=False, max_workers=8):
        if lmu_data_root is None:
            lmu_data_root = LMUDataRoot()
        prepared_root = osp.join(lmu_data_root, cls.PREPARED_DIRNAME)
        if not force:
            try:
                _, manifest = cls.validate_prepared(
                    prepared_root, expected_rows=cls.HF_SPLIT_SIZES[cls.HF_SPLIT]
                )
                return manifest
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                pass

        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError as exc:
            raise ImportError(
                'Preparing SpatialMQA requires huggingface_hub; model processes do not.'
            ) from exc

        annotation_path = hf_hub_download(
            repo_id=cls.HF_REPO_ID,
            filename=cls.HF_SPLIT_FILE,
            repo_type='dataset',
            revision=cls.HF_REVISION,
        )
        annotation_sha256 = cls._sha256(annotation_path)
        if annotation_sha256 != cls.HF_SPLIT_SHA256:
            raise ValueError(
                f'Pinned SpatialMQA source checksum is {annotation_sha256}; '
                f'expected {cls.HF_SPLIT_SHA256}.'
            )
        with open(annotation_path, 'r', encoding='utf-8') as stream:
            image_names = sorted({
                str(json.loads(line).get('image', '')).strip()
                for line in stream if line.strip()
            })
        allow_patterns = [osp.join('images', image_name) for image_name in image_names]
        snapshot_root = snapshot_download(
            repo_id=cls.HF_REPO_ID,
            repo_type='dataset',
            revision=cls.HF_REVISION,
            allow_patterns=allow_patterns,
            max_workers=max_workers,
        )
        return cls._publish_snapshot(
            annotation_path=annotation_path,
            snapshot_root=snapshot_root,
            lmu_data_root=lmu_data_root,
            expected_rows=cls.HF_SPLIT_SIZES[cls.HF_SPLIT],
        )

    def load_data(self, dataset):
        if dataset not in self.supported_datasets():
            raise ValueError(f'Unsupported SpatialMQA dataset name: {dataset}')
        prepared_root = osp.join(LMUDataRoot(), self.PREPARED_DIRNAME)
        data, _ = self.validate_prepared(
            prepared_root, expected_rows=self.HF_SPLIT_SIZES[self.HF_SPLIT]
        )
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_paths = toliststr(line['image_path'])
        if len(image_paths) != 1:
            raise ValueError(f'SpatialMQA expects exactly one image, got {len(image_paths)}')
        if not osp.isfile(image_paths[0]):
            raise FileNotFoundError(f'SpatialMQA image does not exist: {image_paths[0]}')

        options = self._coerce_options(line['options'])
        choices = '\n'.join(
            f'{string.ascii_uppercase[index]}. {option}'
            for index, option in enumerate(options)
        )
        prompt = (
            f'Question: {str(line["question"]).strip()}\n'
            f'Choices:\n{choices}\n'
            'Respond with only the letter of the correct choice.'
        )
        return [
            {'type': 'image', 'value': image_paths[0]},
            {'type': 'text', 'value': prompt},
        ]

    @classmethod
    def _parse_prediction(cls, prediction, options):
        if cls._is_missing(prediction):
            return None
        options = cls._coerce_options(options)
        text = str(prediction).strip()
        if not text:
            return None

        candidates = set()
        valid_letters = string.ascii_uppercase[:len(options)]

        whole_letter = re.fullmatch(r'\s*[\(\[]?([A-Fa-f])[\)\]]?[\s\.!]*', text)
        if whole_letter is not None:
            letter = whole_letter.group(1).upper()
            if letter in valid_letters:
                candidates.add(valid_letters.index(letter))

        marker_pattern = re.compile(
            r'(?i)(?:answer|choice|option)\s*(?:is\s*)?[:=\-]?\s*[\(\[]?([A-F])(?:[\)\]]|\b)'
        )
        for match in marker_pattern.finditer(text):
            letter = match.group(1).upper()
            if letter in valid_letters:
                candidates.add(valid_letters.index(letter))

        for match in re.finditer(r'(?<![A-Za-z])([A-F])(?![A-Za-z])', text):
            letter = match.group(1)
            if letter in valid_letters:
                candidates.add(valid_letters.index(letter))

        normalized_prediction = cls._normalize_text(text)
        for option_index, option in enumerate(options):
            normalized_option = cls._normalize_text(option)
            escaped_option = re.escape(normalized_option).replace(r'\ ', r'\s+')
            option_pattern = re.compile(
                rf'(?<!\w){escaped_option}(?!\w)'
            )
            if option_pattern.search(normalized_prediction):
                candidates.add(option_index)

        if len(candidates) != 1:
            return None
        return string.ascii_uppercase[candidates.pop()]

    @staticmethod
    def _normalized_indices(series):
        indices = []
        for value in series:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                raise ValueError('SpatialMQA evaluation index is missing')
            indices.append(str(value).strip())
        return indices

    def evaluate(self, eval_file, **judge_kwargs):
        predictions = load(eval_file)
        if not isinstance(predictions, pd.DataFrame):
            predictions = pd.DataFrame(predictions)
        missing_columns = {'index', 'prediction'} - set(predictions.columns)
        if missing_columns:
            raise ValueError(f'SpatialMQA predictions are missing columns: {sorted(missing_columns)}')

        canonical = self.data[
            ['index', 'question', 'options', 'answer', 'relation', 'category', 'split', 'source_row']
        ].copy()
        canonical['index'] = self._normalized_indices(canonical['index'])
        canonical_options = []
        for row_number, row in canonical.iterrows():
            options = self._coerce_options(row['options'])
            gold, _, relation = self._canonicalize_answer(options, row['answer'])
            if gold != str(row['answer']).strip().upper():
                raise ValueError(f'Non-canonical SpatialMQA gold at evaluation row {row_number}')
            if self._normalize_text(relation) != self._normalize_text(row['relation']):
                raise ValueError(f'SpatialMQA relation mismatch at evaluation row {row_number}')
            canonical_options.append(options)
        canonical['options'] = canonical_options
        predictions = predictions[['index', 'prediction']].copy()
        predictions['index'] = self._normalized_indices(predictions['index'])
        predictions['_prediction_present'] = True

        for label, frame in [('canonical data', canonical), ('predictions', predictions)]:
            duplicate_mask = frame['index'].duplicated(keep=False)
            if duplicate_mask.any():
                duplicates = frame.loc[duplicate_mask, 'index'].tolist()
                raise ValueError(f'SpatialMQA {label} has duplicate indices: {duplicates[:5]}')

        canonical_indices = set(canonical['index'])
        prediction_indices = set(predictions['index'])
        missing = sorted(canonical_indices - prediction_indices)
        unknown = sorted(prediction_indices - canonical_indices)
        if unknown:
            raise ValueError(
                'SpatialMQA predictions contain unknown canonical indices: '
                f'{unknown[:5]} ({len(unknown)} total)'
            )

        result = canonical.merge(
            predictions, on='index', how='left', sort=False, validate='one_to_one'
        )
        result['parsed_answer'] = [
            self._parse_prediction(prediction, options)
            for prediction, options in zip(result['prediction'], result['options'])
        ]
        result['hit'] = (result['parsed_answer'] == result['answer']).astype(int)
        prediction_present = [
            not self._is_missing(value) and bool(value)
            for value in result.pop('_prediction_present')
        ]
        result['prediction_status'] = [
            'missing' if not present else ('invalid' if parsed is None else 'valid')
            for present, parsed in zip(prediction_present, result['parsed_answer'])
        ]
        result.rename(columns={'answer': 'gold_answer'}, inplace=True)

        overall_fraction = float(result['hit'].mean())
        score = {
            'split': [self.HF_SPLIT],
            'Overall': [100.0 * overall_fraction],
            'Overall_fraction': [overall_fraction],
            'total': [len(result)],
            'valid_predictions': [int((result['prediction_status'] == 'valid').sum())],
            'missing_predictions': [len(missing)],
            'invalid_predictions': [int((result['prediction_status'] == 'invalid').sum())],
        }
        for category in ('x', 'y', 'z'):
            subset = result[result['category'] == category]
            fraction = float(subset['hit'].mean())
            score[category] = [100.0 * fraction]
            score[f'{category}_fraction'] = [fraction]
        for relation in self.RELATION_ORDER:
            subset = result[result['relation'].map(self._normalize_text) == relation]
            fraction = float(subset['hit'].mean())
            score[relation] = [100.0 * fraction]
            score[f'{relation}_fraction'] = [fraction]
        score = pd.DataFrame(score)

        result_file = get_intermediate_file_path(eval_file, '_spatialmqa_result', 'csv')
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(result, result_file)
        dump(score, score_file)
        return score
