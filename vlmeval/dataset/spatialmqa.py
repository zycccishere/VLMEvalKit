import os
import string
from pathlib import Path

import pandas as pd

from .image_mcq import ImageMCQDataset
from ..smp.file import load


class SpatialMQADataset(ImageMCQDataset):
    """Thin adapter for the official SpatialMQA zero-shot protocol."""

    TYPE = 'MCQ'
    MODALITY = 'IMAGE'
    FORCE_DATASET_PROMPT = True
    DATASET_URL = {'SpatialMQA': ''}
    DATASET_MD5 = {}

    @staticmethod
    def _official_question_text(question):
        # HF test differs from the author repository only by this fixed wording
        # on 748 rows; options, answers, images, and row order are identical.
        return str(question).strip().replace('picture', 'image')

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        images = self.dump_image(line)
        options = [
            str(line[letter]).strip()
            for letter in string.ascii_uppercase
            if letter in line and not pd.isna(line[letter])
        ]
        prompt = (
            'You are currently a senior expert in spatial relation reasoning.\n'
            'Given an Image, a Question and Options, your task is to answer the correct spatial '
            'relation. Note that you only need to choose one option from the all options without '
            'explaining any reason.\n'
            # The official scripts replace their literal <image> marker inside
            # the model wrapper. Here the preceding structured image item is
            # that marker, so retaining the text token would duplicate it.
            f'Input: Image: provided above, Question: {self._official_question_text(line["question"])}, '
            f'Options: {"; ".join(options)}.\nOutput:'
        )
        return [
            *[{'type': 'image', 'value': image} for image in images],
            {'type': 'text', 'value': prompt},
        ]

    @staticmethod
    def _normalize_prediction(value):
        return str(value).strip().rstrip('.').strip().casefold()

    def evaluate(self, eval_file, **judge_kwargs):
        del judge_kwargs
        path = Path(eval_file)
        predictions = load(str(path))
        predictions = pd.DataFrame(predictions).copy()
        if predictions['index'].astype(str).duplicated().any():
            raise ValueError('SpatialMQA predictions contain duplicate indices.')
        metadata = self.data.copy()
        metadata['index'] = metadata['index'].astype(str)
        metadata = metadata.set_index('index')

        predictions['index'] = predictions['index'].astype(str)
        unknown = sorted(set(predictions['index']) - set(metadata.index))
        if unknown:
            raise ValueError(f'SpatialMQA predictions contain unknown indices: {unknown[:5]}')
        prediction_rows = {
            str(row['index']): row for row in predictions.to_dict('records')
        }

        details = []
        for index, meta in metadata.iterrows():
            row = prediction_rows.get(index, {'index': index, 'prediction': ''})
            if 'relation' in meta and not pd.isna(meta['relation']):
                relation = str(meta['relation'])
            else:
                relation = str(meta[str(meta['answer']).strip().upper()])
            gold = relation.strip().casefold()
            prediction = self._normalize_prediction(row.get('prediction', ''))
            hit = bool(prediction) and (prediction in gold or gold in prediction)
            detail = dict(row)
            detail['index'] = index
            detail['prediction'] = row.get('prediction', '')
            detail['gold_relation'] = relation
            detail['parse_status'] = 'valid' if index in prediction_rows else 'missing'
            detail['hit'] = int(hit)
            details.append(detail)

        details = pd.DataFrame(details)
        hits = details['hit'].tolist()
        total = len(details)
        accuracy = sum(hits) / total if total else 0.0
        summary = pd.DataFrame([{
            'Overall': accuracy,
            'correct': int(sum(hits)),
            'total': total,
            'predictions': len(predictions),
            'missing': int((details['parse_status'] == 'missing').sum()),
        }])
        details.to_csv(path.with_name(f'{path.stem}_detail.csv'), index=False)
        summary.to_csv(path.with_name(f'{path.stem}_acc.csv'), index=False)
        return summary


SpatialMQA = SpatialMQADataset
