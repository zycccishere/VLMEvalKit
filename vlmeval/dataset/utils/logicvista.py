import pandas as pd
import re
from ...smp import *
from .prompt_tail import tail_tokens_for_judge


FAIL_MSG = 'Failed to obtain answer via API.'


def _extract_boxed_content(text):
    matches = re.findall(r'\\boxed\{([^{}]+)\}', str(text))
    if matches:
        return matches[-1].strip()
    return None


def _parse_logicvista_choice(text):
    raw = str(text).strip()
    if not raw:
        return None

    candidates = []
    boxed = _extract_boxed_content(raw)
    if boxed:
        candidates.append(boxed)

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines[-5:]):
        candidates.append(line)
        lowered = line.lower()
        if lowered.startswith('answer') or lowered.startswith('final answer'):
            parts = re.split(r'[:：]', line, maxsplit=1)
            if len(parts) == 2:
                candidates.append(parts[1].strip())
    candidates.append(raw)

    seen = set()
    normalized_candidates = []
    for cand in candidates:
        cand = str(cand).strip().strip('*').strip().strip('$').strip()
        if cand and cand not in seen:
            normalized_candidates.append(cand)
            seen.add(cand)

    for cand in normalized_candidates:
        digit_match = re.fullmatch(r'[\(\[]?\s*([1-9])\s*[\)\]]?', cand)
        if digit_match:
            return digit_match.group(1)

        if re.fullmatch(r'[A-Za-z]+', cand):
            return ''.join(ch.upper() for ch in cand if ch.isalpha())

        alpha_tokens = re.findall(r'(?<![A-Za-z0-9])([A-Za-z])(?![A-Za-z0-9])', cand)
        if alpha_tokens:
            return ''.join(ch.upper() for ch in alpha_tokens)

        digit_tokens = re.findall(r'(?<!\d)([1-9])(?!\d)', cand)
        if len(digit_tokens) == 1:
            return digit_tokens[0]

    return None


def _digit_to_letter(token):
    token = str(token).strip()
    if len(token) == 1 and token.isdigit():
        idx = int(token)
        if 1 <= idx <= 26:
            return chr(ord('A') + idx - 1)
    return token


def _normalize_logicvista_key(text, answer_hint=None):
    raw = str(text).strip()
    if not raw:
        return ''
    chars = [ch for ch in raw if ch.isalnum()]
    if not chars:
        return ''

    hint = ''.join(ch for ch in str(answer_hint or '') if ch.isalnum())
    hint_is_alpha = bool(hint) and all(ch.isalpha() for ch in hint)
    hint_is_digit = bool(hint) and all(ch.isdigit() for ch in hint)

    normalized = []
    for ch in chars:
        if ch.isdigit():
            normalized.append(_digit_to_letter(ch) if hint_is_alpha else ch)
        elif ch.isalpha():
            if hint_is_digit and len(ch) == 1 and 'A' <= ch.upper() <= 'Z':
                normalized.append(str(ord(ch.upper()) - ord('A') + 1))
            else:
                normalized.append(ch.upper())

    normalized.sort()
    return ''.join(ch.lower() for ch in normalized)


def build_prompt_logicvista(line):
    question = line['question']
    prediction = tail_tokens_for_judge(line['prediction'], max_tokens=96)
    tmpl = (
        "You are a information extractor that extracts multiple choice letter answer choices "
        "or digit answer choices from a paragraph that contains the selected option and sometimes an explanation.\n"
        "Return only the final selected option token(s), such as A, BD, or 3.\n"
        "If the answer does not correspond to a valid option choice, respond with Z.\n"
        'Example 1: \n'
        'Question: <start>\nWhat is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n<end>\n'
        'Answer: <start>\na cute teddy bear\n<end>\nYour output: A\n'
        'Example 2: \n'
        'Question: <start>\nWhat is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n<end>\n'
        'Answer: <start>\nSpider\n<end>\nYour output: Z\n'
        'Example 3: \n'
        'Question: <start>\nWhich figure is a rotation of the object?\n<end>\n'
        'Answer: <start>\nThe figure on the right, labeled "D," is a rotation of the object shown in the top left corner.\n<end>\nYour output: D\n'
        'Example 4: \n'
        'Question: <start>\nWhich of the boxes comes next in the sequence? Select from A-E\n<end>\n'
        'Answer: <start>\nThe sequence of the boxes is A, B, C, D, E.\n<end>\nYour output: ABCDE\n'
        'Example 5: \n'
        'Question: <start>\nWhich is the correct answer according to the image? Select from 1-5\n<end>\n'
        'Answer: <start>\n\\boxed{{3}}\n<end>\nYour output: 3\n'
        'Example 6: \n'
        'Question: <start>\n{}\n<end>\nAnswer tail (last 96 tokens): <start>\n{}\n<end>\nYour output: '
    )

    return tmpl.format(question, prediction)


def LogicVista_auxeval(model, line):
    prompt = build_prompt_logicvista(line)
    log = ''
    retry = 5

    for i in range(retry):
        res = model.generate(prompt, temperature=i * 0.5)
        answer = _normalize_logicvista_key(line['answer'], answer_hint=line['answer'])

        if FAIL_MSG in res:
            log += f'Try {i}: judge output is {res}, failed to parse.\n'
        else:
            parsed = _parse_logicvista_choice(res)
            if parsed is None:
                log += f'Try {i}: judge output is {res}, failed to parse.\n'
                continue
            log += 'Succeed'
            hit = 0
            extracted = _normalize_logicvista_key(parsed, answer_hint=line['answer'])
            if extracted == answer:
                hit = 1
            return dict(log=log, res=parsed, hit=hit)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res='', hit=0)


def evaluate_logicvista(file_path):
    df = load(file_path)

    tot = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    acc = defaultdict(lambda: 0)

    df_tot = df

    df_inductive = df[df["skill"].str.contains("inductive")]
    df_deductive = df[df["skill"].str.contains("deductive")]
    df_numerical = df[df["skill"].str.contains("numerical")]
    df_spatial = df[df["skill"].str.contains("spatial")]
    df_mechanical = df[df["skill"].str.contains("mechanical")]

    tot_correct = df_tot["hit"].sum()
    tot_acc = (tot_correct / df_tot.shape[0]) * 100
    tot['Overall'] = df_tot.shape[0]
    hit['Overall'] = tot_correct
    acc['Overall'] = tot_acc

    inductive_correct = df_inductive["hit"].sum()
    inductive_acc = (inductive_correct / df_inductive.shape[0]) * 100
    tot["inductive"] = df_inductive.shape[0]
    hit["inductive"] = inductive_correct
    acc["inductive"] = inductive_acc

    deductive_correct = df_deductive["hit"].sum()
    deductive_acc = (deductive_correct / df_deductive.shape[0]) * 100
    tot["deductive"] = df_deductive.shape[0]
    hit["deductive"] = deductive_correct
    acc["deductive"] = deductive_acc

    numerical_correct = df_numerical["hit"].sum()
    numerical_acc = (numerical_correct / df_numerical.shape[0]) * 100
    tot["numerical"] = df_numerical.shape[0]
    hit["numerical"] = numerical_correct
    acc["numerical"] = numerical_acc

    spatial_correct = df_spatial["hit"].sum()
    spatial_acc = (spatial_correct / df_spatial.shape[0]) * 100
    tot["spatial"] = df_spatial.shape[0]
    hit["spatial"] = spatial_correct
    acc["spatial"] = spatial_acc

    mechanical_correct = df_mechanical["hit"].sum()
    mechanical_acc = (mechanical_correct / df_mechanical.shape[0]) * 100
    tot["mechanical"] = df_mechanical.shape[0]
    hit["mechanical"] = mechanical_correct
    acc["mechanical"] = mechanical_acc

    res = defaultdict(list)
    for k in tot.keys():
        res['Task&Skill'].append(k)
        res['tot'].append(tot[k])
        res['hit'].append(hit[k])
        res['acc'].append(acc[k])
    res = pd.DataFrame(res)
    return res
