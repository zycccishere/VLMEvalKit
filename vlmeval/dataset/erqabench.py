import json
import pandas as pd

from ..smp.misc import toliststr
from ..smp.file import load
from .image_vqa import ImageVQADataset


class ERQABench(ImageVQADataset):
    """
    ERQA.

    Reference:
      Gemini Robotics: Bringing AI into the Physical World
      https://arxiv.org/html/2503.20020v1
    """

    DATASET_URL = {
        'ERQA': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/ERQA.tsv',
    }
    DATASET_MD5 = {
        'ERQA': '40d45d4f0bb1852dbc68b28ee70d4ca5',
    }

    def _task_category(self):
        return [
            'Action Reasoning',
            'Multi-view Reasoning',
            'Pointing',
            'Spatial Reasoning',
            'State Estimation',
            'Task Reasoning',
            'Trajectory Reasoning',
            'Other'
        ]

    def build_prompt(self, line):
        """
        Images are interleaved into the question according to visual_indices.
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        # Resolve image paths
        if self.meta_only:
            tgt_paths = toliststr(line['image'])
        else:
            tgt_paths = self.dump_image(line)

        question = line['question']
        vis_raw = line.get('visual_indices', [])

        # Parse visual_indices → List[int]
        if isinstance(vis_raw, str):
            try:
                visual_indices = json.loads(vis_raw)
            except Exception:
                parts = [p.strip() for p in vis_raw.strip('[]').split(',') if p.strip()]
                visual_indices = [int(p) for p in parts] if parts else []
        elif isinstance(vis_raw, (list, tuple)):
            visual_indices = list(vis_raw)
        elif pd.isna(vis_raw):
            visual_indices = []
        else:
            visual_indices = []

        # Mismatch: put all images at the beginning
        if len(visual_indices) != len(tgt_paths):
            visual_indices = [0] * len(tgt_paths)

        img_idx_pairs = list(zip(tgt_paths, visual_indices))
        img_idx_pairs.sort(key=lambda x: x[1])

        msgs = []

        # All images first, then full question
        if not visual_indices or all(idx == 0 for idx in visual_indices):
            for p, _ in img_idx_pairs:
                msgs.append(dict(type='image', value=p))
            msgs.append(dict(type='text', value=question))
            return msgs

        # Interleave text and images by character index
        last_pos = 0
        for p, idx in img_idx_pairs:
            if idx == 0:
                msgs.append(dict(type='image', value=p))
            else:
                if idx <= len(question):
                    seg = question[last_pos:idx]
                    if seg:
                        msgs.append(dict(type='text', value=seg))
                    msgs.append(dict(type='image', value=p))
                    last_pos = idx
                else:
                    msgs.append(dict(type='image', value=p))

        if last_pos < len(question):
            seg = question[last_pos:]
            if seg:
                msgs.append(dict(type='text', value=seg))

        if not msgs:
            for p, _ in img_idx_pairs:
                msgs.append(dict(type='image', value=p))
            msgs.append(dict(type='text', value=question))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import eval_mcq_score, build_mcq_score_fn

        # Select MCQ scoring function (rule-based or LLM-based) according to judge_kwargs['model'].
        score_fn = build_mcq_score_fn(**judge_kwargs)

        return eval_mcq_score(
            load_fn=load,
            eval_file=eval_file,
            score_fn=score_fn,
            group_col='question_type',
            order=self._task_category(),
            dataset_name=getattr(self, 'dataset_name', 'ERQA')
        )


class ERQASingleImage(ERQABench):
    """Single-image subset of ERQA (EASI leaderboard TSV).

    ERQA contains 287 single-image rows and 113 multi-image rows (2-16 images;
    all 37 Multi-view Reasoning rows are multi-image). This subset keeps only
    the single-image rows for image-prompt replay studies, following the same
    mechanism as MMMUSingleImageDataset (image_mcq.py): the canonical ERQA.tsv
    is reused and filtered in post_build, so no extra TSV file is stored.
    """

    DATASET_URL = {
        'ERQA_SINGLE': ERQABench.DATASET_URL['ERQA'],
        'ERQA_SINGLE_IMAGE': ERQABench.DATASET_URL['ERQA'],
    }

    DATASET_MD5 = {
        'ERQA_SINGLE': ERQABench.DATASET_MD5['ERQA'],
        'ERQA_SINGLE_IMAGE': ERQABench.DATASET_MD5['ERQA'],
    }

    SOURCE_DATASET = 'ERQA'

    def load_data(self, dataset):
        # Reuse the canonical ERQA TSV while exposing a filtered replay-safe alias.
        original_name = self.dataset_name
        self.dataset_name = self.SOURCE_DATASET
        try:
            return self.prepare_tsv(self.DATASET_URL[dataset], self.DATASET_MD5.get(dataset))
        finally:
            self.dataset_name = original_name

    @staticmethod
    def _is_present(value):
        if isinstance(value, list):
            return len(value) > 0
        return not pd.isna(value)

    @staticmethod
    def _single_image_row(row):
        image_count = 0
        if 'image_path' in row and ERQASingleImage._is_present(row['image_path']):
            image_count = max(image_count, len(toliststr(row['image_path'])))
        if 'image' in row and ERQASingleImage._is_present(row['image']):
            image_count = max(image_count, len(toliststr(row['image'])))
        return image_count == 1

    def post_build(self, dataset):
        keep = self.data.apply(self._single_image_row, axis=1)
        self.data = self.data[keep].copy().reset_index(drop=True)


class ERQAMultiImageAt8(ERQASingleImage):
    """Multi-image ERQA rows (2-16 images), each repeated 8 times ("@8").

    The complement of ERQA_SINGLE: the 113 rows with at least two images, kept
    in their native ERQABench interleaving (visual_indices) and expanded into
    8 replicate rows each, so one native run yields all 904 generations and the
    native ERQABench score is the mean accuracy over every replicate (not
    pass@8). ``index = source_index * 10 + repeat_index``; ``source_index``,
    ``repeat_index`` and ``sample_id`` (``ERQA-<source>-rNN``) keep the
    per-question pairing explicit. Replicates share the source row's image
    files via ``image_path``, so their visual input is byte-identical.
    """

    DATASET_URL = {'ERQA_MULTI_IMAGE': ERQABench.DATASET_URL['ERQA']}
    DATASET_MD5 = {'ERQA_MULTI_IMAGE': ERQABench.DATASET_MD5['ERQA']}
    REPEATS = 8

    @staticmethod
    def _image_count(row):
        image_count = 0
        for col in ('image_path', 'image'):
            if col in row and ERQASingleImage._is_present(row[col]):
                image_count = max(image_count, len(toliststr(row[col])))
        return image_count

    def post_build(self, dataset):
        multi = self.data[self.data.apply(self._image_count, axis=1) >= 2]
        rows = []
        for _, row in multi.iterrows():
            source = int(row['index'])
            images = toliststr(row['image'])
            for repeat in range(self.REPEATS):
                copy = row.copy()
                copy['index'] = source * 10 + repeat
                copy['source_index'] = source
                copy['repeat_index'] = repeat
                copy['sample_id'] = f'ERQA-{source}-r{repeat:02d}'
                copy['image_path'] = [f'{source}_{i}.jpg' for i in range(len(images))]
                rows.append(copy)
        self.data = pd.DataFrame(rows).reset_index(drop=True)
