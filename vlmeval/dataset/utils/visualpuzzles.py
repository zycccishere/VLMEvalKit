import ast
import re

from ...smp import *
from ...utils import track_progress_rich
from .prompt_tail import tail_tokens_for_judge

Option_list = ['A', 'B', 'C', 'D']
FAIL_MSG = 'Failed to obtain answer via API.'


def _normalize_answer_text(ans):
    ans = str(ans).strip()
    ans = ans.replace('（', '(').replace('）', ')')
    ans = ans.replace('【', '[').replace('】', ']')
    return ans


def _answer_tail(ans, max_chars=256):
    ans = _normalize_answer_text(ans)
    return ans[-max_chars:] if len(ans) > max_chars else ans


def _parse_option_map(opt_str):
    raw = opt_str
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(k).strip().upper(): str(v) for k, v in raw.items() if str(k).strip().upper() in Option_list}
    if isinstance(raw, float) and pd.isna(raw):
        return {}

    items = None
    if isinstance(raw, (list, tuple)):
        items = [str(x) for x in raw]
    else:
        text = str(raw).strip()
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return {
                str(k).strip().upper(): str(v)
                for k, v in parsed.items()
                if str(k).strip().upper() in Option_list
            }
        if isinstance(parsed, (list, tuple)):
            items = [str(x) for x in parsed]
        elif text:
            quoted = re.findall(r"""['"]([^'"]+)['"]""", text)
            if quoted:
                items = quoted

    if not items:
        return {}
    return {Option_list[i]: items[i] for i in range(min(len(items), len(Option_list)))}


def _format_option_block(opt_str):
    choices = _parse_option_map(opt_str)
    if len(choices) < 2:
        return ''
    return '\n'.join([f'{key}. {value}' for key, value in choices.items()])


def extract_answer(ans):
    ans = _normalize_answer_text(ans)
    lines = [line.strip() for line in ans.splitlines() if line.strip()]
    last_line = lines[-1] if lines else ans.strip()
    tail = _answer_tail('\n'.join(lines[-2:]) if lines else ans)

    patterns = [
        r"\\boxed\{\s*([A-D])\s*\}\s*$",
        r"(?:final\s+answer|answer)\s*[:：]\s*(?:\*\*)?[\(\[]?\s*([A-D])\s*[\)\]]?(?:\*\*)?\s*$",
        r"^\s*[\(\[]?\s*([A-D])\s*[\)\].,:;!?]?\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, last_line, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    matches = re.findall(r"\\boxed\{\s*([A-D])\s*\}\s*$", tail, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    return "Z"


def _build_visualpuzzles_extract_prompt(item):
    question = str(item.get('question', '')).strip()
    option_block = _format_option_block(item.get('options'))
    prediction_tail = tail_tokens_for_judge(item.get('prediction', ''), max_tokens=96)

    prompt = (
        'You extract the final multiple-choice letter selected by a model.\n'
        'Return exactly one uppercase letter among A, B, C, D, or Z.\n'
        'If the answer does not clearly select a single option, return Z.\n'
        'Example 1:\n'
        'Answer: (C)\nOutput: C\n'
        'Example 2:\n'
        'The only figure Michael cannot make is option (B).\nOutput: B\n'
        'Example 3:\n'
        'I am not sure which option is correct.\nOutput: Z\n'
        f'Question: <start>\n{question}\n'
    )
    if option_block:
        prompt += f'Choices:\n{option_block}\n'
    prompt += f'</start>\nAnswer tail (last 96 tokens): <start>\n{prediction_tail}\n<end>\nOutput: '
    return prompt


def _visualpuzzles_extract_with_judge(model, item):
    prompt = _build_visualpuzzles_extract_prompt(item)
    for retry in range(3):
        ans = model.generate(prompt, temperature=retry * 0.2)
        opt = extract_answer(ans)
        if FAIL_MSG in ans:
            continue
        if opt != 'Z' or ans.strip().upper() == 'Z':
            return dict(opt=opt, log=ans)
    return dict(opt='Z', log='Failed to extract via judge.')


def _annotate_visualpuzzles_predictions(data, model=None, fallback_cache_file=None, nproc=4):
    data = data.copy()
    local_opts = []
    local_sources = []
    local_logs = []
    unresolved = []
    unresolved_keys = []

    for i in range(len(data)):
        row = data.iloc[i]
        local_opt = extract_answer(row['prediction'])
        local_opts.append(local_opt)
        local_logs.append('local_regex')
        if local_opt != 'Z':
            local_sources.append('local_regex')
            continue

        if model is None:
            local_sources.append('local_regex_unresolved')
            continue

        item = {
            'question': row.get('question', ''),
            'prediction': row.get('prediction', ''),
            'options': row.get('options', None),
        }
        unresolved.append(item)
        unresolved_keys.append(row['index'])
        local_sources.append('judge_pending')

    cache = {}
    if fallback_cache_file and osp.exists(fallback_cache_file):
        cache = load(fallback_cache_file)
    cache = {k: v for k, v in cache.items() if isinstance(v, dict) and 'opt' in v}

    pending = []
    pending_keys = []
    for key, item in zip(unresolved_keys, unresolved):
        if key not in cache:
            pending.append(dict(model=model, item=item))
            pending_keys.append(key)

    if pending:
        results = track_progress_rich(
            _visualpuzzles_extract_with_judge,
            pending,
            nproc=nproc,
            chunksize=max(1, min(nproc, 8)),
            save=fallback_cache_file,
            keys=pending_keys,
        )
        for key, result in zip(pending_keys, results):
            cache[key] = result

    final_opts = []
    final_sources = []
    final_logs = []
    for i in range(len(data)):
        row = data.iloc[i]
        local_opt = local_opts[i]
        if local_opt != 'Z':
            final_opts.append(local_opt)
            final_sources.append('local_regex')
            final_logs.append(local_logs[i])
            continue

        cache_result = cache.get(row['index'])
        if cache_result and cache_result.get('opt'):
            final_opts.append(cache_result['opt'])
            final_sources.append('judge_extract')
            final_logs.append(cache_result.get('log', ''))
        else:
            final_opts.append('Z')
            final_sources.append('local_regex_unresolved')
            final_logs.append(local_logs[i])

    data['extracted_answer'] = final_opts
    data['extract_source'] = final_sources
    data['extract_log'] = final_logs
    return data


def prepare_visualpuzzles_eval_file(eval_file, storage, tmp_file, model=None, nproc=4):
    data = load(eval_file)
    data = _annotate_visualpuzzles_predictions(
        data,
        model=model,
        fallback_cache_file=tmp_file,
        nproc=nproc,
    )
    dump(data, storage)


def VisulPuzzles_acc(result_file, model=None, fallback_cache_file=None, save_file=None, nproc=4):
    categories = [
        'overall',
        'spatial',
        'algorithmic',
        'analogical',
        'inductive',
        'deductive',
    ]
    difficulties = [
        'easy',
        'medium',
        'hard',
        'overall'
    ]

    data = load(result_file)
    if 'extracted_answer' not in data or model is not None:
        data = _annotate_visualpuzzles_predictions(
            data,
            model=model,
            fallback_cache_file=fallback_cache_file,
            nproc=nproc,
        )
        if save_file is not None:
            dump(data, save_file)
    lt = len(data)
    hit = defaultdict(lambda: 0)
    tot = defaultdict(lambda: 0)
    from tqdm import tqdm
    for i in tqdm(range(lt)):
        item = data.iloc[i]
        cate = item['category']
        tot['overall'] += 1
        tot[cate] += 1

        pred_answer = item['extracted_answer'] if 'extracted_answer' in item else extract_answer(item['prediction'])
        if pred_answer.lower() == item['answer'].lower():
            hit['overall'] += 1
            hit[cate] += 1

    res = defaultdict(list)

    for k in categories:
        res['category'].append(k)
        res['tot'].append(tot[k])
        res['hit'].append(hit[k])
        res['acc'].append(hit[k] / tot[k] * 100 if tot[k] > 0 else 0.0)

    hit_level = defaultdict(lambda: 0)
    tot_level = defaultdict(lambda: 0)
    res_level = defaultdict(list)
    for i in tqdm(range(lt)):
        item = data.iloc[i]
        level = item['difficulty']
        tot_level['overall'] += 1
        tot_level[level] += 1

        pred_answer = item['extracted_answer'] if 'extracted_answer' in item else extract_answer(item['prediction'])
        if pred_answer.lower() == item['answer'].lower():
            hit_level['overall'] += 1
            hit_level[level] += 1

    for k in difficulties:
        res_level['level'].append(k)
        res_level['tot'].append(tot_level[k])
        res_level['hit'].append(hit_level[k])
        res_level['acc'].append(hit_level[k] / tot_level[k] * 100 if tot_level[k] > 0 else 0.0)

    return res, res_level
