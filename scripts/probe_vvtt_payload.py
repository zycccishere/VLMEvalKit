"""Audit native wrapper + vLLM renderer inputs without loading model weights."""
import argparse
from collections import Counter
import hashlib
import json
import random
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(args):
    from vlmeval.cli.run_benchmark import BenchmarkRunner
    from vlmeval.dataset import build_dataset
    from vlmeval.probes.standard_entry_parity import (
        build_standard_prompt, load_light_model, patched_environ)
    from vllm.config import ModelConfig, VllmConfig
    from vllm.renderers.hf import HfRenderer
    from vllm.tokenizers import cached_tokenizer_from_config
    import yaml

    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit_matrix = args.output.parent/'native-preview-matrix.yaml'
    matrix = yaml.safe_load(args.matrix.read_text())
    matrix['results_root'] = str(args.output.parent/'native-preview')
    audit_matrix.write_text(yaml.safe_dump(matrix,allow_unicode=True,sort_keys=False))
    options = SimpleNamespace(matrix_config=audit_matrix, model_config=args.models_config,
        task_manifest=args.tasks,nodes=1,node_rank=0,gpu_ids='0',models='',policies='',
        modes='',transforms='',datasets='',scheduler='',manifest_is_node_shard=False,
        resume_infer=None,plan_only=True)
    runner = BenchmarkRunner(ROOT, options)
    records=[]
    for key in args.models:
        family = {'gemma3_12b':'gemma3','qwen25vl_32b':'qwen25vl'}[key]
        tasks = [t for t in runner.tasks if t.model_key==key]
        env=runner.build_env(runner.models[key],tasks[0],['0'])
        env.update(REPLAY_DUMP_DIR='',REPLAY_TRACE_LEVEL='off',REPLAY_STAGE_DEBUG='0')
        sys.modules.pop('vlmeval.config_text_filler_minimal',None)
        model=load_light_model(env,runner.models[key].registry_name,family)
        assert Path(model.model_path).resolve()==Path(runner.models[key].model_path).resolve()
        mc=ModelConfig(model=model.model_path,tokenizer=model.model_path,trust_remote_code=True,
                       max_model_len=32768,limit_mm_per_prompt={'image':4})
        renderer=HfRenderer(VllmConfig(model_config=mc),cached_tokenizer_from_config(mc))
        for task in tasks:
            with patched_environ(env):
                dataset=build_dataset(task.dataset)
                for _,row in dataset.data.iloc[:args.samples].iterrows():
                    base=build_standard_prompt(model,dataset,task.dataset,row)
                    pair={}
                    for label,mode in [('slp','image_text_image_text'),('vvtt','image_image_text_text')]:
                        model.replay_cfg['mode']=mode
                        random.seed(20261108)
                        if family=='gemma3':
                            content=model._apply_replay_pipeline(base,dataset=task.dataset)
                            payload=model._message_to_vllm_payload(content,dataset=task.dataset)
                            signature=[(x['type'],x.get('value')) for x in content]
                        else:
                            content=model._prepare_content_vllm(base,dataset=task.dataset)
                            signature=[(x['type'],x.get('text') if x['type']=='text' else x.get('image')) for x in content]
                            payload=model._build_vllm_request(base,dataset=task.dataset)
                        processed=renderer.render_cmpl([payload])[0]
                        ids=processed['prompt_token_ids']
                        spans=processed['mm_placeholders']['image']
                        images=payload['multi_modal_data']['image']
                        assert len(images)==len(spans)==2
                        pair[label]=dict(token_count=len(ids),token_ids=ids,signature=signature,
                            image_hashes=[hashlib.sha256(image.tobytes()).hexdigest() for image in images],
                            image_sizes=[list(image.size) for image in images],
                            spans=[dict(offset=x.offset,length=x.length) for x in spans])
                    a,b=pair['slp'],pair['vvtt']
                    assert Counter(a['signature'])==Counter(b['signature'])
                    assert a['image_hashes']==b['image_hashes'] and a['image_sizes']==b['image_sizes']
                    assert [s['length'] for s in a['spans']]==[s['length'] for s in b['spans']]
                    gap=lambda p:p['spans'][1]['offset']-p['spans'][0]['offset']-p['spans'][0]['length']
                    assert gap(b)<gap(a), 'VVTT did not move V2 before the first question'
                    result=dict(model=key,dataset=task.dataset,index=str(row['index']),
                        slp_tokens=a['token_count'],vvtt_tokens=b['token_count'],
                        delta=b['token_count']-a['token_count'],token_multiset_equal=Counter(a['token_ids'])==Counter(b['token_ids']),
                        slp_image_spans=a['spans'],vvtt_image_spans=b['spans'],
                        same_images_and_text=True)
                    records.append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
                    args.output.parent.mkdir(parents=True,exist_ok=True)
                    args.output.write_text(json.dumps(dict(records=records,all_lengths_equal=all(r['delta']==0 for r in records)),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix',type=Path,required=True);p.add_argument('--models-config',type=Path,required=True)
    p.add_argument('--tasks',type=Path,required=True);p.add_argument('--models',nargs='+',default=['gemma3_12b','qwen25vl_32b'])
    p.add_argument('--samples',type=int,default=1);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
