"""Normal LogicVista generation using the native vLLM engine and a scoped mask."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

from vlmeval.probes.readout_v2 import write_json, sha256_file
from vlmeval.probes.standard_entry_parity import build_standard_prompt, repo_git_snapshot
from vlmeval.vlm.qwen2_vl.prompt import Qwen2VLPromptMixin
from vlmeval.vlm.replay_policy import apply_replay


def native_greedy_params(args):
    """Reuse the actual vLLM entry, not the HF/LMDeploy constructor defaults."""
    from vlmeval.vlm.qwen2_vl.model import Qwen2VLChatReplay
    base = Qwen2VLChatReplay._build_vllm_sampling_params(
        SimpleNamespace(max_new_tokens=args.max_tokens))
    expected = dict(temperature=0.0, top_p=1.0, repetition_penalty=1.0,
                    presence_penalty=0.0, frequency_penalty=0.0, min_p=0.0)
    for key, value in expected.items():
        if getattr(base, key) != value:
            raise ValueError(f'Native greedy contract overridden: {key}={getattr(base, key)}')
    if base.max_tokens != args.max_tokens or base.stop_token_ids or base.stop:
        raise ValueError('Unexpected native output-budget/stop override')
    params = []
    for i in range(args.limit):
        sample = base.clone()
        sample.seed = args.seed + args.offset + i
        sample.logprobs = args.logprobs or None
        params.append(sample)
    fields = [*expected, 'top_k', 'max_tokens', 'stop', 'stop_token_ids',
              'ignore_eos', 'skip_special_tokens', 'spaces_between_special_tokens']
    effective = {key: getattr(base, key) for key in fields}
    if hasattr(base, 'watermarking'):
        effective['watermarking'] = base.watermarking
    effective.update(logprobs=args.logprobs or None,
                     sample_seeds=[p.seed for p in params], source='Qwen2VLChatReplay._build_vllm_sampling_params')
    return params, effective


def prepare(args):
    from transformers import AutoProcessor
    from vlmeval.dataset.image_vqa import LogicVista
    from vlmeval.config_qwen_minimal import supported_VLM
    # Read the existing entry's prompt/preprocessing defaults without loading
    # a second model. LogicVista's paper entry uses the DATASET prompt.
    defaults = supported_VLM['Qwen2VLChatReplay'].keywords
    args.min_pixels = defaults['min_pixels']
    args.max_pixels = defaults['max_pixels']
    args.use_custom_prompt = defaults['use_custom_prompt']
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    dataset = LogicVista('LogicVista')
    prompt_builder = Qwen2VLPromptMixin(use_custom_prompt=args.use_custom_prompt)
    prompt_builder.set_dump_image(dataset.dump_image)
    rows = dataset.data.iloc[args.offset:args.offset+args.limit]
    prepared = []
    for _, row in rows.iterrows():
        base = build_standard_prompt(prompt_builder,dataset,'LogicVista',row)
        if [item['type'] for item in base] != ['image','text']:
            raise ValueError('LogicVista intervention requires one image followed by question text')
        per = dict(index=str(row['index']), answer=str(row['answer']), prompts={})
        for condition, mode in [('slp','image_text_image_text'),('vtt','image_text_text')]:
            items = apply_replay(base,mode,repeat_times=1)
            content, images = [], []
            for item in items:
                if item['type']=='image':
                    with Image.open(item['value']) as im:
                        images.append(im.convert('RGB').copy())
                    content.append(dict(type='image',image=item['value']))
                elif item['type']=='text':
                    content.append(dict(type='text',text=item['value']))
                else:
                    raise ValueError(item['type'])
            text = processor.apply_chat_template([dict(role='user',content=content)],
                                                 tokenize=False,add_generation_prompt=True)
            per['prompts'][condition] = dict(prompt=text,multi_modal_data={'image':images})
        prepared.append(per)
    ids = dict(image_id=processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'),
               start_id=processor.tokenizer.convert_tokens_to_ids('<|vision_start|>'),
               end_id=processor.tokenizer.convert_tokens_to_ids('<|vision_end|>'))
    return prepared, ids


def run(args):
    conditions = args.conditions.split(',')
    valid = {'slp', 'all_visible', 'no_write', 'vtt'}
    if not conditions or not set(conditions) <= valid or len(set(conditions)) != len(conditions):
        raise ValueError('Conditions must be unique members of slp,all_visible,no_write,vtt')
    if min(args.limit, args.tp, args.batch_size, args.max_tokens) <= 0 or args.offset < 0:
        raise ValueError('Counts must be positive and offset nonnegative')
    if 'all_visible' in conditions and args.logprobs < 1:
        raise ValueError('Technical parity requires first-token logprobs')
    from vllm import LLM
    from vlmeval.probes.generation_attention import image_bounds
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        raise FileExistsError('Use a fresh output directory; do not overwrite a run')
    rows, token_ids = prepare(args)
    if len(rows) != args.limit:
        raise ValueError(f'Requested {args.limit} samples, found {len(rows)}; check offset/limit')
    params, effective_sampling = native_greedy_params(args)
    source_root = Path(__file__).resolve().parents[2]
    import vllm.v1.attention.backends.flex_attention as flex_backend
    manifest = dict(args=vars(args),kind='normal_generation_attention',dataset='LogicVista',
        dataset_sha256=sha256_file(Path(os.environ['LMUData'])/'LogicVista.tsv'),
        versions={p:importlib.metadata.version(p) for p in ['torch','transformers','vllm']},
        repo=repo_git_snapshot(Path(__file__).resolve().parents[2]),sampling=effective_sampling,
        native_qwen_config_sha256=sha256_file(Path(__file__).resolve().parents[1]/'config_qwen_minimal.py'),
        source_sha256={name:sha256_file(source_root/name) for name in [
            'vlmeval/probes/generation_attention.py', 'vlmeval/probes/run_generation_attention.py',
            'vlmeval/probes/standard_entry_parity.py', 'vlmeval/probes/readout_v2.py',
            'vlmeval/probes/score_generation_attention.py',
            'vlmeval/vlm/replay_policy.py', 'vlmeval/vlm/qwen2_vl/model.py',
            'vlmeval/config_qwen_minimal.py', 'vlmeval/dataset/image_vqa.py',
            'vlmeval/dataset/utils/logicvista.py']},
        flex_backend_sha256=sha256_file(Path(flex_backend.__file__)),
        model_metadata_sha256={p.name:sha256_file(p) for p in Path(args.model).glob('*.json')},
        weight_files=[dict(name=p.name,size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                      for p in sorted(Path(args.model).glob('*.safetensors'))],
        sample_indices=[r['index'] for r in rows],token_ids=token_ids,
        attention_scope='all decoder layers; V2 including vision delimiters cannot read V1/T1; T2/decode remain causal',
        controls='all_visible is a technical control, not an edge-matched Sham')
    for name, digest in manifest['source_sha256'].items():
        destination = out/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root/name, destination)
        if sha256_file(destination) != digest:
            raise RuntimeError(f'Source changed while freezing run: {name}')
    write_json(out/'manifest.json',manifest)
    if args.prepare_only:
        return
    llm = LLM(model=args.model, dtype='bfloat16', tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.max_batched_tokens or args.max_model_len,enable_prefix_caching=False,
        enable_chunked_prefill=True,async_scheduling=False,enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,attention_config={'backend':'FLEX_ATTENTION'},
        mm_processor_kwargs={'min_pixels':args.min_pixels,'max_pixels':args.max_pixels},
        limit_mm_per_prompt={'image':2},seed=args.seed,
        worker_extension_cls='vlmeval.probes.generation_attention.GenerationAttentionWorker')
    print(llm.collective_rpc('install_generation_attention',args=(token_ids,)),flush=True)
    outputs = {}
    for condition in conditions:
        mask = 'native' if condition in ['slp','vtt'] else condition
        llm.collective_rpc('set_generation_intervention',args=(mask,))
        requests = [r['prompts']['vtt' if condition=='vtt' else 'slp'] for r in rows]
        t0 = time.monotonic()
        result = llm.generate(requests,params,use_tqdm=True)
        elapsed = time.monotonic()-t0
        audit = llm.collective_rpc('generation_intervention_audit')
        write_json(out/f'{condition}.raw.json',dict(audit=audit,records=[dict(
            index=s['index'],request_id=r.request_id,prompt_token_ids=r.prompt_token_ids,
            prediction=r.outputs[0].text,token_ids=r.outputs[0].token_ids)
            for s,r in zip(rows,result)]))
        records = []
        for sample, response in zip(rows,result):
            ids = list(response.prompt_token_ids)
            digest = hashlib.sha256(np.asarray(ids,dtype=np.int32).tobytes()).hexdigest()
            bounds = image_bounds(ids,**token_ids) if condition!='vtt' else None
            if condition in ['all_visible','no_write']:
                for worker in audit:
                    # vLLM 0.31 InputProcessor.assign_request_id appends an
                    # 8-hex nonce internally; RequestOutput returns the external
                    # ID. Validate the exact versioned format AND token hash.
                    hits = [a for a in worker['requests'] if re.fullmatch(
                        re.escape(response.request_id)+r'-[0-9a-f]{8}',a['request_id'])]
                    assert len(hits)==1,(response.request_id,worker)
                    a = hits[0]
                    assert a['prompt_sha256']==digest
                    assert [a[k] for k in ['key_start','query_start','query_end']]==[
                        bounds.key_start,bounds.query_start,bounds.query_end]
                    assert a['removed_edges']==(bounds.removed_edges if condition=='no_write' else 0)
                    assert a['v2_query_tokens_seen']==bounds.query_end-bounds.query_start
            prediction = response.outputs[0]
            records.append(dict(index=sample['index'],answer=sample['answer'],
                request_id=response.request_id,prediction=prediction.text,
                token_ids=list(prediction.token_ids),prompt_tokens=len(ids),
                prompt_sha256=digest,bounds=vars(bounds) if bounds else None,
                first_logprobs={str(k):v.logprob for k,v in (prediction.logprobs[0] if prediction.logprobs else {}).items()},
                finish_reason=prediction.finish_reason))
        outputs[condition] = records
        write_json(out/f'{condition}.json',dict(seconds=elapsed,records=records,audit=audit))
        print(json.dumps(dict(condition=condition,n=len(records),seconds=elapsed,
            output_tokens=sum(len(r['token_ids']) for r in records))),flush=True)
    if 'slp' in outputs and 'all_visible' in outputs:
        for native, control in zip(outputs['slp'],outputs['all_visible']):
            assert native['prompt_sha256']==control['prompt_sha256']
            assert native['token_ids']==control['token_ids'],('technical-control generation mismatch',native['index'])
            assert set(native['first_logprobs'])==set(control['first_logprobs'])
            assert max(abs(native['first_logprobs'][k]-control['first_logprobs'][k])
                       for k in native['first_logprobs']) < 1e-3
        write_json(out/'parity.json',dict(passed=True,samples=len(rows),
            criterion='identical generated tokens; top-5 first-token logprobs max absolute difference < 1e-3'))
    write_json(out/'generation-complete.json',dict(conditions=list(outputs),samples=len(rows),
               scoring_status='not_scored'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='/data/local/yicheng/temporary/curated/prompt-replay/models/Qwen2.5-VL-32B-Instruct')
    p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=4)
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--tp',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--max-model-len',type=int,default=16384)
    p.add_argument('--max-batched-tokens',type=int)
    p.add_argument('--max-tokens',type=int,default=2048)
    p.add_argument('--seed',type=int,default=20261109)
    p.add_argument('--temperature',type=float,default=0.0,choices=[0.0])
    p.add_argument('--logprobs',type=int,default=5,choices=range(0,21))
    p.add_argument('--gpu-memory-utilization',type=float,default=.85)
    p.add_argument('--conditions',default='slp,all_visible,no_write,vtt')
    p.add_argument('--prepare-only',action='store_true')
    run(p.parse_args())


if __name__=='__main__':
    main()
