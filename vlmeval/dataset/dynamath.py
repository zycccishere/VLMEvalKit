import re
import json
import numpy as np
import pandas as pd
import sys
import math
import os
import os.path as osp
import argparse

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from .utils.prompt_tail import tail_tokens_for_judge
from ..utils import track_progress_rich
from ..smp import load, dump, d2df, toliststr
from ..smp.file import get_intermediate_file_path


def preprocess(str1):
    str1 = str(str1).strip()
    str1 = str1.replace("\\n", "\n")
    if str1.startswith("```"):
        lines = [x for x in str1.splitlines() if not x.strip().startswith("```")]
        str1 = "\n".join(lines).strip()
    return str1


def _extract_boxed_content(text):
    matches = re.findall(r'\\+boxed\s*\{([^{}]+)\}', text)
    if matches:
        return matches[-1].strip()
    return None


def _iter_balanced_brace_chunks(text):
    chunks = []
    start = None
    depth = 0
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == '{':
            if depth == 0:
                start = idx
            depth += 1
            continue

        if ch == '}':
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                chunks.append(text[start: idx + 1])
                start = None

    return chunks


def _extract_short_answer_from_json_like(text):
    text = str(text)
    text = text.replace('\\{', '{').replace('\\}', '}')
    cands = [text]
    cands.extend(reversed(_iter_balanced_brace_chunks(text)))
    for cand in cands:
        try:
            obj = json.loads(cand, strict=False)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        for key in ("short answer", "short_answer", "answer"):
            if key in obj and obj[key] is not None:
                return str(obj[key]).strip()
    return None


def transfer(str1):
    if "\u03c0" in str1:
        strs = str1.split('\u03c0')
        str1 = strs[0]
        return float(str1) * np.pi
    else:
        return float(str1)


def _normalize_text_answer(text):
    return re.sub(r'[^a-z0-9]+', '', str(text).strip().lower())


def _text_answers_match(pred_answer, gold_answer):
    pred_norm = _normalize_text_answer(pred_answer)
    gold_norm = _normalize_text_answer(gold_answer)

    if not pred_norm or not gold_norm:
        return False
    if pred_norm == gold_norm:
        return True
    if pred_norm in gold_norm or gold_norm in pred_norm:
        return True
    return False


def parse_answer(answer, answer_type="multiple choice"):
    answer = str(answer).strip()
    boxed = _extract_boxed_content(answer)
    if boxed is not None:
        answer = boxed

    if answer_type == "float":
        answer = answer.replace(",", "")
        m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\s*\u03c0)?', answer)
        if m is None:
            return False, None
        token = m.group(0).replace(" ", "")
        try:
            if "\u03c0" in token:
                return True, transfer(token)
            return True, float(token)
        except Exception:
            return False, None
    elif answer_type == "multiple choice":
        letters = re.findall(r'[A-Z]', answer.upper())
        if len(set(letters)) == 1 and len(letters) >= 1:
            return True, letters[0]
        return False, None
    else:
        return True, answer


def _extract_final_answer_line(text):
    text = preprocess(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-8:]):
        stripped = line.strip().strip('*').strip()
        lowered = stripped.lower()
        if lowered.startswith('final answer'):
            parts = re.split(r'[:：]', stripped, maxsplit=1)
            return parts[1].strip() if len(parts) == 2 else stripped
        if lowered.startswith('answer'):
            parts = re.split(r'[:：]', stripped, maxsplit=1)
            return parts[1].strip() if len(parts) == 2 else stripped
    return None


def _extract_safe_short_answer_candidate(pred):
    pred = preprocess(pred)
    short_answer = _extract_short_answer_from_json_like(pred)
    if short_answer is not None:
        return short_answer

    boxed = _extract_boxed_content(pred)
    if boxed is not None:
        boxed_json = _extract_short_answer_from_json_like(boxed)
        return boxed_json if boxed_json is not None else boxed

    final_line = _extract_final_answer_line(pred)
    if final_line is not None:
        final_json = _extract_short_answer_from_json_like(final_line)
        return final_json if final_json is not None else final_line

    return None


def DynaMath_local_parse(line):
    pred = line['prediction']
    pred = preprocess(pred)

    try:
        short_answer = _extract_safe_short_answer_candidate(pred)
        if short_answer is None:
            raise ValueError('No safe local short-answer candidate found.')
        succeed, short_answer = parse_answer(short_answer, answer_type=line['answer_type'])
        assert succeed
    except:
        return False, None, pred
    return True, short_answer, pred


def DynaMath_auxeval(model, line):
    succeed, short_answer, pred = DynaMath_local_parse(line)

    if not succeed:
        # Failed to parse the JSON, use an auxiliary LLM to get the short answer
        if line['answer_type'] == 'multiple choice':
            inst = (
                "Output only the final selected choice letter in a single line, "
                "such as 'A', 'B', 'C', or 'F'. If no valid choice is selected, output Z."
            )
        elif line['answer_type'] == 'float':
            inst = "Output only the final short answer as a single floating-point number in one line."
        else:
            inst = (
                "Output only the final short answer in a single line. "
                "Any float numbers in the answer should be formatted as three-digit floating-point numbers."
            )

        pred_tail = tail_tokens_for_judge(pred, max_tokens=96)
        prompt = (
            "Extract the final short answer from the response below.\n"
            f"Instruction: {inst}\n"
            f"Free-form answer tail (last 96 tokens): {pred_tail}\n"
            "Output:"
        )
        response = model.generate(prompt)
        succeed, short_answer = parse_answer(response, line['answer_type'])

    if line['answer_type'] == 'float':
        if succeed:
            diff = float(short_answer) - float(line['answer'])
            if abs(diff) <= 0.001:
                return dict(parse=True, extracted=short_answer, correct=True)
            else:
                return dict(parse=True, extracted=short_answer, correct=False)
        else:
            return dict(parse=False, extracted=None, correct=False)
    elif line['answer_type'] == 'multiple choice':
        if succeed:
            return dict(parse=True, extracted=short_answer, correct=(short_answer == line['answer']))
        else:
            return dict(parse=False, extracted=None, correct=False)
    else:
        if succeed:
            return dict(parse=True, extracted=short_answer, correct=_text_answers_match(short_answer, line['answer']))
        else:
            return dict(parse=False, extracted=None, correct=False)


class Dynamath(ImageBaseDataset):

    TYPE = 'VQA'
    DATASET_URL = {
        # 'DynaMath': 'https://opencompass.openxlab.space/utils/VLMEval/DynaMath.tsv',
        'DynaMath': '/datasets/vlmeval/DynaMath.tsv',
        'DynaMath_noprompt': 'https://opencompass.openxlab.space/utils/VLMEval/DynaMath.tsv',
        'DynaMathSmall': '/datasets/small_benchmarks/DynaMath_small.tsv'
    }
    DATASET_MD5 = {
        'DynaMath': 'b8425ad9a7114571fc9366e013699494',
        'DynaMath_noprompt': 'b8425ad9a7114571fc9366e013699494',
        'DynaMathSmall': 'dd1b1ab7cd1bb7d9021f1758544ebd9e'
    }
    GUIDE = """
## Answer Instruction Please provide an answer to the question outlined above. Your response should adhere \
to the following JSON format, which includes one key: 'short answer'. {INST}

Example of expected JSON response format:

"""
    EXAMPLE = {"short answer": "[Concise Answer]"}
    LEGACY_EXAMPLE = {
        "solution": "[Detailed step-by-step explanation]",
        "short answer": "[Concise Answer]",
    }
    TEXT_EXAMPLE = json.dumps(EXAMPLE, indent=4)
    LEGACY_TEXT_EXAMPLE = json.dumps(LEGACY_EXAMPLE, indent=4)

    @staticmethod
    def _dynamath_prompt_schema():
        # The standard replay entry explicitly sets legacy_two_keys only for
        # Qwen2.5-VL DynaMath table reproduction. Other callers default to the
        # answer-only schema so non-Qwen routes do not silently inherit Qwen's
        # legacy public-table prompt.
        schema = os.environ.get('DYNAMATH_PROMPT_SCHEMA', 'short_answer_only').strip().lower()
        if schema not in {'short_answer_only', 'legacy_two_keys'}:
            raise ValueError(f'Unsupported DYNAMATH_PROMPT_SCHEMA: {schema}')
        return schema

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        prompt = f"## Question\n {line['question']}"
        if line['answer_type'] == 'multiple choice':
            inst = "Provide the corresponing choice option in the 'short answer' key, such as 'A', 'B', 'C', or 'D'."
        elif line['answer_type'] == 'float':
            inst = "Format the answer as a three-digit floating-point number and provide it in the 'short answer' key."
        else:
            inst = "Float numbers in the answer should be formatted as three-digit floating-point numbers."

        if 'noprompt' not in self.dataset_name:
            # In directly-answer template mode, avoid extra JSON-schema instructions.
            # This keeps dataset prompt compatible with global answer-only templating.
            if os.environ.get('REPLAY_PROMPT_TEMPLATE_NAME', '').strip().lower() == 'directly_answer':
                if line['answer_type'] == 'multiple choice':
                    direct_inst = "Answer with only the corresponding choice option, such as 'A', 'B', 'C', or 'D'."
                elif line['answer_type'] == 'float':
                    direct_inst = "Answer with only a three-digit floating-point number."
                else:
                    direct_inst = "Answer with a short phrase only. Any float number should use three-digit floating-point format."
                prompt = prompt + f"\n## Answer Instruction\n{direct_inst}"
            else:
                schema = self._dynamath_prompt_schema()
                if schema == 'legacy_two_keys':
                    legacy_guide = self.GUIDE.replace("one key: 'short answer'", "two keys: 'solution' and 'short answer'. The 'solution' key can contain detailed steps needed to solve the question, and the 'short answer' key should provide a concise response")
                    prompt = prompt + legacy_guide.format(INST=inst) + self.LEGACY_TEXT_EXAMPLE
                else:
                    prompt = prompt + self.GUIDE.format(INST=inst) + self.TEXT_EXAMPLE

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        judge_name = judge_kwargs.pop('model', 'gpt-4o-mini')

        model = build_judge(model=judge_name, **judge_kwargs)

        storage = get_intermediate_file_path(eval_file, f'_{judge_name}')
        score_file = get_intermediate_file_path(eval_file, f'_{judge_name}_score', 'csv')
        tmp_file = get_intermediate_file_path(eval_file, f'_{judge_name}', 'pkl')
        nproc = judge_kwargs.pop('nproc', 6)  # noqa: F841

        res = load(tmp_file) if os.path.exists(tmp_file) else {}
        res = {k: v for k, v in res.items() if v is not None}

        model.system_prompt = """\
You are a helpful assistant that helps me to format free-form answers into a short answer according to the instruction.
"""
        if not osp.exists(storage):
            data = load(eval_file)
            lt = len(data)
            pending_indices = [i for i in range(lt) if data.iloc[i]['index'] not in res]
            requires_judge = any(not DynaMath_local_parse(data.iloc[i])[0] for i in pending_indices)
            if requires_judge:
                assert model.working(), 'DynaMath evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE
            payloads = [dict(model=model, line=data.iloc[i]) for i in pending_indices]
            keys = [data.iloc[i]['index'] for i in pending_indices]

            if len(keys):
                results = track_progress_rich(DynaMath_auxeval, payloads, nproc=nproc, save=tmp_file, keys=keys)
                for k, r in zip(keys, results):
                    res[k] = r

            data['parse'] = [res[idx]['parse'] for idx in data['index']]
            data['extracted'] = [res[idx]['extracted'] for idx in data['index']]
            data['correct'] = [res[idx]['correct'] for idx in data['index']]
            dump(data, storage)

        data = load(storage)
        # Calculate Average Accuracy
        score_avg = {}
        score_avg['Overall'] = np.mean(data['correct'])

        subs = set(data['subject'])
        for sub in subs:
            data_sub = data[data['subject'] == sub]
            score_avg[f'Subject-{sub}'] = np.mean(data_sub['correct'])

        lvls = set(data['knowledge_level'])
        for lvl in lvls:
            data_lvl = data[data['knowledge_level'] == lvl]
            score_avg[f'Level-{lvl}'] = np.mean(data_lvl['correct'])

        # Calculate the Worst Case Accuracy
        score_worst = {}
        data_worst = data[data['varid'] == 1]
        qid2corr = {idx: True for idx in data_worst['index']}
        lt = len(data)
        for i in range(lt):
            item = data.iloc[i]
            qid2corr[item['qid']] *= item['correct']
        data_worst['correct'] = [qid2corr[idx] for idx in data_worst['qid']]
        score_worst['Overall'] = np.mean(data_worst['correct'])

        subs = set(data_worst['subject'])
        for sub in subs:
            data_sub = data_worst[data_worst['subject'] == sub]
            score_worst[f'Subject-{sub}'] = np.mean(data_sub['correct'])

        lvls = set(data_worst['knowledge_level'])
        for lvl in lvls:
            data_lvl = data_worst[data_worst['knowledge_level'] == lvl]
            score_worst[f'Level-{lvl}'] = np.mean(data_lvl['correct'])

        d1 = {'Setting': 'Average'}
        d1.update(score_avg)
        d2 = {'Setting': 'Worst Case'}
        d2.update(score_worst)
        score = pd.concat([d2df(d1), d2df(d2)], ignore_index=True)

        dump(score, score_file)
        return score
