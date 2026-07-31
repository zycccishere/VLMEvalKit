import hashlib
import json
import math
import os
import os.path as osp
import struct
from dataclasses import dataclass
from numbers import Real
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .image_base import ImageBaseDataset
from ..smp import dump, load
from ..smp.file import LMUDataRoot, get_intermediate_file_path


HF_DATASET_ID = 'JierunChen/Ref-L4'
HF_REVISION = 'd62c4f4e5f3a639b34adab34e5c8ffeb39f168c1'
UPSTREAM_CODE_REVISION = 'bae942c852b52c5be1d8214c978e984294793b91'
PREPARED_SCHEMA_VERSION = 1
PREPARED_DIRNAME = osp.join('RefL4', HF_REVISION)
PREPARED_TSV = 'RefL4.tsv'
PREPARED_MANIFEST = 'manifest.json'

EXPECTED_SPLIT_COUNTS = {'val': 13420, 'test': 31921}
EXPECTED_TOTAL_ROWS = sum(EXPECTED_SPLIT_COUNTS.values())
REQUIRED_PREPARED_COLUMNS = (
    'index',
    'annotation_id',
    'question',
    'image_path',
    'bbox_x',
    'bbox_y',
    'bbox_w',
    'bbox_h',
    'bbox_area',
    'bbox_id',
    'ori_category_id',
    'image_id',
    'height',
    'width',
    'is_rewrite',
    'split',
    'source_revision',
)

COORDINATE_SPACE = 'normalized_1000'
COORDINATE_SIZE = (1000.0, 1000.0)
IOU_THRESHOLDS = (0.5, 0.75, 0.9)
MACC_THRESHOLDS = tuple(i / 100 for i in range(50, 100, 5))


class RefL4PreparationError(RuntimeError):
    pass


class BBoxProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedBBox:
    bbox_xyxy: Tuple[float, float, float, float]
    coordinate_space: str
    coordinate_size: Tuple[float, float]


def prepared_root(data_root: Optional[str] = None) -> str:
    root = data_root or LMUDataRoot()
    return osp.join(root, PREPARED_DIRNAME)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _format_ids(values: Iterable[str], limit: int = 5) -> str:
    items = sorted(str(value) for value in values)
    head = ', '.join(items[:limit])
    if len(items) > limit:
        head += f', ... ({len(items)} total)'
    return head


def _as_finite_numbers(values: object, length: int, field: str) -> Tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise BBoxProtocolError(f'`{field}` must contain exactly {length} numbers.')

    parsed = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise BBoxProtocolError(f'`{field}` must contain only numbers.')
        number = float(value)
        if not math.isfinite(number):
            raise BBoxProtocolError(f'`{field}` contains a non-finite coordinate.')
        parsed.append(number)
    return tuple(parsed)


def validate_xyxy(values: object, field: str = 'bbox_2d') -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = _as_finite_numbers(values, 4, field)
    if x2 <= x1 or y2 <= y1:
        raise BBoxProtocolError(f'`{field}` must be a non-degenerate [x1, y1, x2, y2] box.')
    return x1, y1, x2, y2


def xywh_to_xyxy(values: object, field: str = 'bbox') -> Tuple[float, float, float, float]:
    x, y, width, height = _as_finite_numbers(values, 4, field)
    if width <= 0 or height <= 0:
        raise BBoxProtocolError(f'`{field}` must have positive width and height.')
    return validate_xyxy((x, y, x + width, y + height), field=field)


def parse_bbox_protocol(text: object) -> ParsedBBox:
    """Parse the exact model-independent normalized-coordinate JSON protocol."""
    if not isinstance(text, str) or not text.strip():
        raise BBoxProtocolError('Prediction is empty or is not text.')

    def reject_duplicate_keys(pairs):
        payload = {}
        for key, value in pairs:
            if key in payload:
                raise BBoxProtocolError(f'Prediction JSON contains duplicate key `{key}`.')
            payload[key] = value
        return payload

    try:
        payload = json.loads(text.strip(), object_pairs_hook=reject_duplicate_keys)
    except BBoxProtocolError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BBoxProtocolError('Prediction must be exactly one JSON object and no other text.') from exc
    if not isinstance(payload, dict):
        raise BBoxProtocolError('Prediction must be exactly one JSON object and no other text.')
    required_keys = {'bbox_2d', 'bbox_format', 'coordinate_space', 'coordinate_size'}
    if set(payload) != required_keys:
        raise BBoxProtocolError(
            f'Prediction JSON keys must be exactly {sorted(required_keys)}.'
        )
    if payload.get('bbox_format') != 'xyxy':
        raise BBoxProtocolError('`bbox_format` must explicitly be `xyxy`.')

    coordinate_space = payload.get('coordinate_space')
    if coordinate_space != COORDINATE_SPACE:
        raise BBoxProtocolError(f'`coordinate_space` must be `{COORDINATE_SPACE}`.')

    coordinate_size = _as_finite_numbers(payload.get('coordinate_size'), 2, 'coordinate_size')
    if coordinate_size != COORDINATE_SIZE:
        raise BBoxProtocolError('`coordinate_size` must be exactly [1000, 1000].')
    bbox_xyxy = validate_xyxy(payload.get('bbox_2d'))

    return ParsedBBox(
        bbox_xyxy=bbox_xyxy,
        coordinate_space=str(coordinate_space),
        coordinate_size=(coordinate_size[0], coordinate_size[1]),
    )


def convert_bbox_to_original_pixels(
    parsed: ParsedBBox,
    original_size: Sequence[float],
) -> Tuple[float, float, float, float]:
    """Convert the fixed normalized-1000 canvas to canonical original pixels.

    The mapping is independent of each model processor's internal resize. No
    coordinate system is guessed from the numerical range.
    """
    original_width, original_height = _as_finite_numbers(original_size, 2, 'original_size')
    if original_width <= 0 or original_height <= 0:
        raise BBoxProtocolError('`original_size` width and height must be positive.')

    source_width, source_height = parsed.coordinate_size
    x1, y1, x2, y2 = parsed.bbox_xyxy

    if parsed.coordinate_space != COORDINATE_SPACE or (source_width, source_height) != COORDINATE_SIZE:
        raise BBoxProtocolError(f'Unsupported coordinate space: {parsed.coordinate_space}')
    scale_x = original_width / source_width
    scale_y = original_height / source_height
    converted = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)

    return validate_xyxy(converted, field='converted_bbox_2d')


def parse_prediction_to_original_pixels(
    text: object,
    original_size: Sequence[float],
) -> Tuple[float, float, float, float]:
    parsed = parse_bbox_protocol(text)
    return convert_bbox_to_original_pixels(parsed, original_size)


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    # Official Ref-L4 constructs torch tensors without a dtype override, so
    # overlap arithmetic is float32. Preserve that boundary behavior without
    # adding a torch or NumPy runtime dependency.
    def float32(value: float) -> float:
        return struct.unpack('f', struct.pack('f', float(value)))[0]

    a = tuple(float32(value) for value in validate_xyxy(box_a, field='box_a'))
    b = tuple(float32(value) for value in validate_xyxy(box_b, field='box_b'))

    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection_width = max(float32(right - left), 0.0)
    intersection_height = max(float32(bottom - top), 0.0)
    intersection = float32(intersection_width * intersection_height)
    area_a = float32(float32(a[2] - a[0]) * float32(a[3] - a[1]))
    area_b = float32(float32(b[2] - b[0]) * float32(b[3] - b[1]))
    union = float32(float32(area_a + area_b) - intersection)
    return float32(intersection / max(union, float32(1e-6)))


def _read_prepared_package(package_root: str, check_images: bool = True) -> Tuple[pd.DataFrame, dict]:
    manifest_path = osp.join(package_root, PREPARED_MANIFEST)
    tsv_path = osp.join(package_root, PREPARED_TSV)
    if not osp.isfile(manifest_path) or not osp.isfile(tsv_path):
        raise FileNotFoundError(
            f'Ref-L4 is not prepared at {package_root}. Run `python scripts/prepare_ref_l4.py` first.'
        )

    with open(manifest_path, 'r', encoding='utf-8') as stream:
        manifest = json.load(stream)

    expected_manifest = {
        'schema_version': PREPARED_SCHEMA_VERSION,
        'source_repo': HF_DATASET_ID,
        'source_revision': HF_REVISION,
        'upstream_code_revision': UPSTREAM_CODE_REVISION,
        'row_count': EXPECTED_TOTAL_ROWS,
        'split_counts': EXPECTED_SPLIT_COUNTS,
        'annotation_file': PREPARED_TSV,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise RefL4PreparationError(
                f'Prepared Ref-L4 manifest has invalid `{key}`: {manifest.get(key)!r}; expected {expected!r}.'
            )

    expected_sha = manifest.get('annotation_sha256')
    if not isinstance(expected_sha, str) or _sha256(tsv_path) != expected_sha:
        raise RefL4PreparationError('Prepared Ref-L4 annotation TSV checksum does not match its manifest.')

    data = pd.read_csv(tsv_path, sep='\t', dtype={'index': str, 'source_revision': str})
    missing_columns = sorted(set(REQUIRED_PREPARED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise RefL4PreparationError(f'Prepared Ref-L4 TSV is missing columns: {missing_columns}')
    if len(data) != EXPECTED_TOTAL_ROWS:
        raise RefL4PreparationError(
            f'Prepared Ref-L4 TSV has {len(data)} rows; expected {EXPECTED_TOTAL_ROWS}.'
        )

    split_counts = data['split'].value_counts().to_dict()
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise RefL4PreparationError(
            f'Prepared Ref-L4 split counts are {split_counts}; expected {EXPECTED_SPLIT_COUNTS}.'
        )
    if data['index'].duplicated().any() or data['annotation_id'].duplicated().any():
        raise RefL4PreparationError('Prepared Ref-L4 annotations contain duplicate annotation ids.')
    annotation_keys = data['annotation_id'].astype(str)
    if list(data['index'].astype(str)) != list(annotation_keys):
        raise RefL4PreparationError('Prepared Ref-L4 `index` does not match canonical annotation_id.')
    if set(int(value) for value in data['annotation_id']) != set(range(EXPECTED_TOTAL_ROWS)):
        raise RefL4PreparationError('Prepared Ref-L4 annotation ids are not the canonical range 0..45340.')
    if set(data['source_revision'].astype(str)) != {HF_REVISION}:
        raise RefL4PreparationError('Prepared Ref-L4 TSV contains an unexpected source revision.')

    relative_images = [str(path) for path in data['image_path']]
    unique_images = set(relative_images)
    file_validation = manifest.get('file_validation')
    expected_file_validation = {
        'expected': len(unique_images),
        'checked': len(unique_images),
        'missing': 0,
        'dimension_mismatches': 0,
    }
    if manifest.get('image_file_count') != len(unique_images):
        raise RefL4PreparationError('Prepared Ref-L4 manifest has an invalid image_file_count.')
    if file_validation != expected_file_validation:
        raise RefL4PreparationError(
            f'Prepared Ref-L4 manifest has invalid file_validation: {file_validation!r}.'
        )
    if manifest.get('bbox_source_format') != 'xywh_original_pixel':
        raise RefL4PreparationError('Prepared Ref-L4 manifest has an invalid bbox_source_format.')

    image_entries = manifest.get('image_files', [])
    if len(image_entries) != len(unique_images):
        raise RefL4PreparationError('Prepared Ref-L4 manifest image inventory count mismatch.')
    inventory_paths = set()
    for entry in image_entries:
        relative_path = str(entry.get('path', ''))
        if osp.isabs(relative_path) or not relative_path.startswith(f'images{os.sep}'):
            raise RefL4PreparationError(f'Prepared Ref-L4 manifest has unsafe image path: {relative_path}')
        resolved = osp.realpath(osp.join(package_root, relative_path))
        if osp.commonpath([osp.realpath(package_root), resolved]) != osp.realpath(package_root):
            raise RefL4PreparationError(f'Prepared Ref-L4 manifest has unsafe image path: {relative_path}')
        inventory_paths.add(resolved)
        if not osp.isfile(resolved):
            raise RefL4PreparationError(f'Prepared Ref-L4 image is missing: {resolved}')
        if osp.getsize(resolved) != entry.get('bytes'):
            raise RefL4PreparationError(f'Prepared Ref-L4 image size mismatch: {resolved}')
        if _sha256(resolved) != entry.get('sha256'):
            raise RefL4PreparationError(f'Prepared Ref-L4 image checksum mismatch: {resolved}')

    real_package_root = osp.realpath(package_root)
    resolved_images = []
    for path in relative_images:
        if osp.isabs(path) or not path.startswith(f'images{os.sep}'):
            raise RefL4PreparationError(f'Prepared Ref-L4 TSV contains an unsafe image_path: {path}')
        resolved = osp.realpath(osp.join(package_root, path))
        if osp.commonpath([real_package_root, resolved]) != real_package_root:
            raise RefL4PreparationError(f'Prepared Ref-L4 TSV contains an unsafe image_path: {path}')
        resolved_images.append(resolved)
    data['image_path'] = resolved_images
    if set(resolved_images) != inventory_paths:
        raise RefL4PreparationError('Prepared Ref-L4 table images do not match manifest inventory.')
    if check_images:
        missing_images = sorted({path for path in data['image_path'] if not osp.isfile(path)})
        if missing_images:
            raise RefL4PreparationError(
                f'Prepared Ref-L4 package is missing image files: {_format_ids(missing_images)}'
            )

    return data, manifest


class RefL4Dataset(ImageBaseDataset):
    """Ref-L4 referring-expression grounding with a dataset-owned protocol."""

    TYPE = 'GROUNDING'
    MODALITY = 'IMAGE'
    FORCE_DATASET_PROMPT = True

    DATASET_SPLITS = {
        'RefL4': 'all',
        'Ref-L4': 'all',
        'RefL4-val': 'val',
        'Ref-L4-val': 'val',
        'RefL4-test': 'test',
        'Ref-L4-test': 'test',
    }

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_SPLITS)

    def load_data(self, dataset):
        if dataset not in self.DATASET_SPLITS:
            raise ValueError(f'Unsupported Ref-L4 dataset name: {dataset}')

        data, self.prepared_manifest = _read_prepared_package(prepared_root(), check_images=True)
        split = self.DATASET_SPLITS[dataset]
        if split != 'all':
            data = data[data['split'] == split].copy()
        return data.reset_index(drop=True)

    def post_build(self, dataset):
        annotation_keys = self.data['annotation_id'].astype(str)
        if annotation_keys.duplicated().any():
            duplicates = annotation_keys[annotation_keys.duplicated(keep=False)]
            raise ValueError(f'Ref-L4 canonical annotations contain duplicate ids: {_format_ids(duplicates)}')
        if list(self.data['index'].astype(str)) != list(annotation_keys):
            raise ValueError('Ref-L4 `index` must be the canonical annotation id.')

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        paths = self.dump_image(line)
        if not isinstance(paths, list) or len(paths) != 1:
            raise ValueError('Ref-L4 requires exactly one source image per annotation.')

        description = str(line['question'])
        prompt = (
            'Locate the single image region described below.\n'
            f'Description: {description}\n\n'
            'Use a fixed 0-to-1000 coordinate canvas: image left/top is 0, '
            'image right/bottom is 1000, independent of image resolution or internal resizing.\n'
            'Return exactly one JSON object and no other text, using this protocol:\n'
            '{"bbox_2d":[x1,y1,x2,y2],"bbox_format":"xyxy",'
            '"coordinate_space":"normalized_1000","coordinate_size":[1000,1000]}\n'
            'All four coordinates must be finite values with x1 < x2 and y1 < y2. '
            'Values outside 0 to 1000 are allowed only when the predicted box extends beyond the image boundary.'
        )
        return [
            {'type': 'image', 'value': paths[0]},
            {'type': 'text', 'value': prompt},
        ]

    def evaluate(self, eval_file, **judge_kwargs):
        predictions = load(eval_file)
        if not isinstance(predictions, pd.DataFrame):
            predictions = pd.DataFrame(predictions)
        required = {'index', 'prediction'}
        missing_columns = sorted(required - set(predictions.columns))
        if missing_columns:
            raise KeyError(f'Ref-L4 prediction file is missing columns: {missing_columns}')

        gold = self.data.copy()
        gold['_annotation_key'] = gold['annotation_id'].astype(str)
        predictions = predictions.copy()
        predictions['_annotation_key'] = predictions['index'].astype(str)

        gold_duplicates = gold.loc[gold['_annotation_key'].duplicated(keep=False), '_annotation_key']
        if len(gold_duplicates):
            raise ValueError(f'Duplicate canonical annotation ids: {_format_ids(gold_duplicates)}')
        pred_duplicates = predictions.loc[
            predictions['_annotation_key'].duplicated(keep=False), '_annotation_key'
        ]
        if len(pred_duplicates):
            raise ValueError(f'Duplicate prediction annotation ids: {_format_ids(pred_duplicates)}')

        gold_keys = set(gold['_annotation_key'])
        prediction_keys = set(predictions['_annotation_key'])
        missing = gold_keys - prediction_keys
        extra = prediction_keys - gold_keys
        if extra:
            raise ValueError(f'Unknown prediction annotation ids: {_format_ids(extra)}')

        prediction_map = predictions.set_index('_annotation_key').to_dict('index')
        details: List[Dict[str, object]] = []
        ious: List[float] = []
        valid_parse_count = 0
        invalid_count = 0

        for row in gold.to_dict('records'):
            annotation_id = row['_annotation_key']
            gt_bbox = xywh_to_xyxy(
                (row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']),
                field='canonical_bbox',
            )

            prediction_record = prediction_map.get(annotation_id)
            missing_prediction = prediction_record is None
            raw_prediction = '' if missing_prediction else prediction_record['prediction']
            pred_bbox = None
            invalid_reason = ''

            if missing_prediction:
                iou = 0.0
            else:
                try:
                    pred_bbox = parse_prediction_to_original_pixels(
                        raw_prediction,
                        original_size=(row['width'], row['height']),
                    )
                except BBoxProtocolError as exc:
                    invalid_reason = str(exc)

                if invalid_reason:
                    invalid_count += 1
                    iou = 0.0
                else:
                    valid_parse_count += 1
                    iou = bbox_iou(gt_bbox, pred_bbox)

            ious.append(iou)
            parse_valid = not missing_prediction and not invalid_reason
            detail = {
                'index': annotation_id,
                'split': row['split'],
                'prediction': raw_prediction,
                'pred_bbox_xyxy': '' if pred_bbox is None else json.dumps(pred_bbox),
                'gt_bbox_xyxy': json.dumps(gt_bbox),
                'parse_valid': int(parse_valid),
                'missing_prediction': int(missing_prediction),
                'invalid_reason': invalid_reason,
                'iou': iou,
            }
            for threshold in IOU_THRESHOLDS:
                detail[f'acc_iou_{threshold}'] = int(iou > threshold)
            details.append(detail)

        sample_count = len(ious)
        if sample_count == 0:
            raise ValueError('Ref-L4 canonical dataset is empty.')
        summary: Dict[str, object] = {
            'Split': self.DATASET_SPLITS.get(self.dataset_name, self.dataset_name),
            'Samples': sample_count,
            'valid_parse_count': valid_parse_count,
            'valid_parse_rate': valid_parse_count / sample_count,
            'missing_count': len(missing),
            'invalid_count': invalid_count,
        }
        for threshold in IOU_THRESHOLDS:
            hits = sum(iou > threshold for iou in ious)
            summary[f'Ann-level acc iou {threshold}'] = hits / sample_count * 100
        threshold_accuracies = [
            sum(iou > threshold for iou in ious) / sample_count
            for threshold in MACC_THRESHOLDS
        ]
        summary['Ann-level macc iou 0.5:0.95'] = (
            sum(threshold_accuracies) / len(threshold_accuracies) * 100
        )

        detail_file = get_intermediate_file_path(eval_file, '_detail', 'csv')
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        detail_frame = pd.DataFrame(details)
        score_frame = pd.DataFrame([summary])
        dump(detail_frame, detail_file)
        dump(score_frame, score_file)
        return score_frame
