#!/usr/bin/env python3
"""
Convert vlmevalkit inference output (xlsx/tsv) to RLHF-V answer jsonl and run MMHal or ObjHal eval.
Usage:
  MMHal: python run_hal_eval_after_infer.py --work-dir ... --model-name ... --dataset MMHal --openai-key ...
  ObjHal: python run_hal_eval_after_infer.py --work-dir ... --model-name ... --dataset ObjHal --coco-annotations /path/to/coco2014/annotations --openai-key ...
"""
import argparse
import json
import os
import subprocess
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VLMEVALKIT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
RLHFV_ROOT = os.environ.get('RLHFV_ROOT') or os.path.join(VLMEVALKIT_ROOT, 'RLHF-V-main')
RLHFV_EVAL = os.path.join(RLHFV_ROOT, 'eval')
RLHFV_DATA = os.path.join(RLHFV_ROOT, 'eval', 'data')


def _env_with_rlhfv():
    env = os.environ.copy()
    env['PYTHONPATH'] = RLHFV_ROOT + os.pathsep + RLHFV_EVAL + os.pathsep + env.get('PYTHONPATH', '')
    # Align HAL eval with other judge paths: default to proxy base unless caller overrides.
    if not env.get('OPENAI_API_BASE_JUDGE'):
        env['OPENAI_API_BASE_JUDGE'] = env.get('OPENAI_API_BASE', 'https://api.openai.com/v1')
    return env


def load_infer_result(infer_path):
    if infer_path.endswith('.xlsx'):
        return pd.read_excel(infer_path)
    if infer_path.endswith('.tsv'):
        return pd.read_csv(infer_path, sep='\t')
    raise ValueError('Unsupported infer file: ' + infer_path)


def get_predictions_ordered(df, pred_col='prediction', fallback_col='detailed_prediction'):
    col = pred_col if pred_col in df.columns else fallback_col
    if col not in df.columns:
        raise ValueError('Need column prediction or detailed_prediction')
    if 'index' in df.columns:
        df = df.sort_values('index')
    return df, [str(x).replace('Assistant:', '').strip() for x in df[col].tolist()]


def write_answer_jsonl(df, predictions, out_path):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    missing_pred = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        # RLHF-V objhal loader requires question/prompt and text/answer fields.
        questions = df['question'].tolist() if 'question' in df.columns else [''] * len(predictions)
        image_ids = df['image_id'].tolist() if 'image_id' in df.columns else [None] * len(predictions)
        for i, pred in enumerate(predictions):
            # Avoid writing literal "nan" when inference output is missing.
            if pd.isna(pred):
                pred_text = ''
                missing_pred += 1
            else:
                pred_text = str(pred).replace('Assistant:', '').strip()
            item = {
                'question': str(questions[i]) if questions[i] is not None else '',
                'text': pred_text,
            }
            if image_ids[i] is not None and str(image_ids[i]).strip() != '':
                item['image_id'] = image_ids[i]
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    if missing_pred:
        print(f'Warning: {missing_pred} predictions are missing in infer result; wrote empty text for them.', flush=True)


def run_mmhal_eval(answer_file, save_dir, api_key, template_file=None):
    template_file = template_file or os.path.join(RLHFV_DATA, 'mmhal-bench_answer_template.json')
    template_out = answer_file + '.template.json'
    eval_out = answer_file + '.mmhal_test_eval.json'
    env = _env_with_rlhfv()
    subprocess.run([sys.executable, os.path.join(RLHFV_EVAL, 'change_mmhal_predict_template.py'),
                    '--response-template', template_file, '--answers-file', answer_file, '--save-file', template_out],
                   check=True, cwd=RLHFV_ROOT, env=env)
    log_file = answer_file + '.eval_log.txt'
    gpt_model = os.environ.get('MMHAL_GPT_MODEL', 'gpt-4o')
    max_retries = os.environ.get('MMHAL_MAX_RETRIES_PER_SAMPLE', '30')
    retry_sleep = os.environ.get('MMHAL_RETRY_SLEEP_SEC', '2')
    with open(log_file, 'w') as lf:
        subprocess.run([sys.executable, '-u', os.path.join(RLHFV_EVAL, 'eval_gpt_mmhal.py'),
                       '--response', template_out, '--evaluation', eval_out, '--api-key', api_key,
                       '--gpt-model', gpt_model,
                       '--max-retries-per-sample', str(max_retries),
                       '--retry-sleep-sec', str(retry_sleep)],
                      check=True, cwd=RLHFV_ROOT, env=_env_with_rlhfv(), stdout=lf, stderr=subprocess.STDOUT)
    merge_out = answer_file + '.mmhal_test_all_infos.json'
    subprocess.run([sys.executable, os.path.join(RLHFV_EVAL, 'merge_mmhal_review_with_predict.py'),
                   '--review_path', eval_out, '--predict_path', answer_file, '--save_path', merge_out],
                  check=True, cwd=RLHFV_ROOT, env=_env_with_rlhfv())
    scores_file = os.path.join(save_dir, 'mmhal_scores.txt')
    with open(scores_file, 'w') as sf:
        subprocess.run([sys.executable, os.path.join(RLHFV_EVAL, 'summarize_gpt_mmhal_review.py'), save_dir],
                      check=True, cwd=RLHFV_ROOT, env=_env_with_rlhfv(), stdout=sf)
    return scores_file


def run_objhal_eval(answer_file, save_dir, coco_path, api_key, org_file=None):
    org_file = org_file or os.path.join(RLHFV_DATA, 'obj_halbench_300_with_image.jsonl')
    env = _env_with_rlhfv()
    gpt_model = os.environ.get('OBJHAL_GPT_MODEL', 'gpt-4o')
    fail_limit = os.environ.get('OBJHAL_GPT_FAIL_LIMIT', '40')
    max_workers = os.environ.get('OBJHAL_GPT_MAX_WORKERS', '16')
    log_file = answer_file + '.eval_log.txt'
    with open(log_file, 'w') as lf:
        subprocess.run([sys.executable, '-u', os.path.join(RLHFV_EVAL, 'eval_gpt_obj_halbench.py'),
                        '--coco_path', coco_path, '--cap_folder', save_dir,
                        '--cap_type', os.path.basename(answer_file), '--org_folder', org_file,
                        '--use_gpt', '--openai_key', api_key,
                        '--gpt_model', gpt_model,
                        '--gpt_fail_limit', str(fail_limit),
                        '--gpt_max_workers', str(max_workers)],
                       check=True, cwd=RLHFV_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    scores_file = os.path.join(save_dir, 'obj_halbench_scores.txt')
    with open(scores_file, 'w') as sf:
        subprocess.run([sys.executable, os.path.join(RLHFV_EVAL, 'summarize_gpt_obj_halbench_review.py'), save_dir],
                      check=True, cwd=RLHFV_ROOT, env=env, stdout=sf)
    return scores_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--model-name', required=True)
    ap.add_argument('--dataset', required=True, choices=['MMHal', 'ObjHal', 'mmhal', 'objhal'])
    ap.add_argument('--openai-key', default=os.environ.get('OPENAI_API_KEY'))
    ap.add_argument('--coco-annotations', default=os.environ.get('COCO2014_ANNOTATIONS'))
    args = ap.parse_args()
    dataset = args.dataset.strip().lower()
    dataset = 'MMHal' if dataset == 'mmhal' else 'ObjHal'

    model_dir = os.path.join(args.work_dir, args.model_name)
    if dataset == 'MMHal':
        infer_file = os.path.join(model_dir, args.model_name + '_MMHal.xlsx')
        if not os.path.isfile(infer_file):
            infer_file = os.path.join(model_dir, args.model_name + '_MMHal.tsv')
        answer_name = 'mmhal-bench_answer.jsonl'
    else:
        infer_file = os.path.join(model_dir, args.model_name + '_ObjHal.xlsx')
        if not os.path.isfile(infer_file):
            infer_file = os.path.join(model_dir, args.model_name + '_ObjHal.tsv')
        answer_name = 'obj_halbench_answer.jsonl'

    if not os.path.isfile(infer_file):
        print('Infer file not found:', infer_file, file=sys.stderr)
        sys.exit(1)

    df = load_infer_result(infer_file)
    df, predictions = get_predictions_ordered(df)
    answer_file = os.path.join(model_dir, answer_name)
    write_answer_jsonl(df, predictions, answer_file)
    save_dir = model_dir

    if dataset == 'MMHal':
        if not args.openai_key:
            print('--openai-key or OPENAI_API_KEY required for MMHal', file=sys.stderr)
            sys.exit(1)
        scores_file = run_mmhal_eval(answer_file, save_dir, args.openai_key)
    else:
        if not args.coco_annotations or not os.path.isdir(args.coco_annotations):
            print('--coco-annotations or COCO2014_ANNOTATIONS required for ObjHal', file=sys.stderr)
            sys.exit(1)
        if not args.openai_key:
            print('--openai-key or OPENAI_API_KEY required for ObjHal', file=sys.stderr)
            sys.exit(1)
        scores_file = run_objhal_eval(answer_file, save_dir, args.coco_annotations, args.openai_key)

    print('Scores:', open(scores_file).read())


if __name__ == '__main__':
    main()
