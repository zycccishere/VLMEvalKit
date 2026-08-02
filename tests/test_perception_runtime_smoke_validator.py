import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'validate_perception_runtime_smoke.py'


def _record(**overrides):
    record = {
        'stage': 'final_model_input',
        'backend': 'vllm',
        'consumer_api': 'vllm.LLM.generate',
        'model_family': 'qwen2.5-vl',
        'task_identity': {
            'dataset': 'CountQA',
            'model_key': 'qwen25vl_3b',
            'condition': 'iq',
            'canonical_index': 'countqa-0001-00',
        },
        'generation_config': {
            'n': 1,
            'best_of': 1,
            'temperature': 0.0,
            'top_p': 1.0,
            'top_k': -1,
            'max_tokens': 32,
            'repetition_penalty': 1.0,
            'presence_penalty': 0.0,
            'frequency_penalty': 0.0,
            'decoding_mode': 'greedy',
            'summary_source_type': 'SamplingParams',
            'summary_completeness': 'selected_effective_fields',
        },
    }
    record.update(overrides)
    return record


def _run(root, expected_records=1):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            '--root',
            str(root),
            '--required-task',
            'CountQA:qwen25vl_3b:iq',
            '--expect-records-per-task',
            str(expected_records),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def _run_for_task(root, task, expected_records=1):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            '--root',
            str(root),
            '--required-task',
            task,
            '--expect-records-per-task',
            str(expected_records),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_runtime_validator_accepts_complete_real_contract(tmp_path):
    dump = tmp_path / 'task' / 'replay_raw.jsonl'
    dump.parent.mkdir()
    dump.write_text(json.dumps(_record()) + '\n', encoding='utf-8')
    result = _run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_runtime_validator_rejects_empty_root(tmp_path):
    result = _run(tmp_path)
    assert result.returncode == 1
    assert 'no generation configs found' in result.stdout


def test_runtime_validator_rejects_wrong_backend_and_candidate_count(tmp_path):
    dump = tmp_path / 'task' / 'replay_raw.jsonl'
    dump.parent.mkdir()
    record = _record(backend='transformers', consumer_api='fake.not_vllm')
    record['generation_config']['n'] = 7
    record['generation_config']['best_of'] = 7
    dump.write_text(json.dumps(record) + '\n', encoding='utf-8')
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "backend='transformers'" in result.stdout
    assert 'generation_config.n=7' in result.stdout
    assert 'best_of=7' in result.stdout


def test_runtime_validator_rejects_duplicate_canonical_index(tmp_path):
    dump = tmp_path / 'task' / 'replay_raw.jsonl'
    dump.parent.mkdir()
    record = _record()
    dump.write_text(
        json.dumps(record) + '\n' + json.dumps(record) + '\n',
        encoding='utf-8',
    )
    result = _run(tmp_path, expected_records=2)
    assert result.returncode == 1
    assert 'duplicate canonical_index' in result.stdout


def test_runtime_validator_rejects_wrong_model_family(tmp_path):
    dump = tmp_path / 'task' / 'replay_raw.jsonl'
    dump.parent.mkdir()
    dump.write_text(json.dumps(_record(model_family='gemma3')) + '\n', encoding='utf-8')
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "model_family='gemma3'" in result.stdout


def test_runtime_validator_rejects_missing_required_task(tmp_path):
    dump = tmp_path / 'task' / 'replay_raw.jsonl'
    dump.parent.mkdir()
    record = _record(
        consumer_api='vllm.LLM.generate',
        model_family='gemma3',
        task_identity={
            'dataset': 'CountQA',
            'model_key': 'gemma3_4b',
            'condition': 'iq',
            'canonical_index': 'countqa-0001-00',
        },
    )
    dump.write_text(json.dumps(record) + '\n', encoding='utf-8')
    result = _run(tmp_path)
    assert result.returncode == 1
    assert 'required task has no final input records' in result.stdout


def test_runtime_validator_rejects_required_task_count_mismatch(tmp_path):
    dump = tmp_path / 'task' / 'replay_raw.jsonl'
    dump.parent.mkdir()
    dump.write_text(json.dumps(_record()) + '\n', encoding='utf-8')
    result = _run(tmp_path, expected_records=2)
    assert result.returncode == 1
    assert 'has 1 final input records, expected 2' in result.stdout


def test_runtime_validator_requires_non_countqa_tasks_too(tmp_path):
    dump = tmp_path / 'task' / 'replay_raw.jsonl'
    dump.parent.mkdir()
    record = _record(
        task_identity={
            'dataset': 'SpatialMQA',
            'model_key': 'qwen25vl_3b',
            'condition': 'iqiq',
            'canonical_index': 'spatialmqa-test-000000',
        },
    )
    count_record = _record()
    dump.write_text(
        json.dumps(record) + '\n' + json.dumps(count_record) + '\n',
        encoding='utf-8',
    )
    result = _run_for_task(tmp_path, 'SpatialMQA:qwen25vl_3b:iqiq')
    assert result.returncode == 0, result.stdout + result.stderr
