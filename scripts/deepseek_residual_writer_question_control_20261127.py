"""Paired question-format controls for an isolated dialogue-derived writer."""
import json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.cache_utils import DynamicCache
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha
from scripts.deepseek_kernel_question_format_20261127 import selected_cases,build_facts,DATA
from scripts.deepseek_residual_kv_writer_20261127 import ResidualWriter

RUN=ROOT/'runs/research/residual-kv-writer-20261127-v1'
OUT=ROOT/'runs/research/residual-writer-question-control-20261127-v1'

def main():
    OUT.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    def save(name,data):
        tmp=OUT/(name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(OUT/name)
    try:
        torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
        paths={'base':MODEL/'model.safetensors','candidate':RUN/'candidate.pt','report':RUN/'result.json','data':DATA}
        hashes={k:sha(v) for k,v in paths.items()};report=json.loads(paths['report'].read_text(encoding='utf8'))
        if hashes['base']!=WEIGHT_SHA or hashes['candidate']!=report['candidate_sha256']:raise ValueError('Unexpected input digest')
        payload=torch.load(paths['candidate'],map_location='cpu',weights_only=True)
        if payload['base_sha256']!=WEIGHT_SHA:raise ValueError('Candidate base mismatch')
        _,values,questions=selected_cases(json.loads(DATA.read_text(encoding='utf8')));facts=build_facts(values)
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        writer=ResidualWriter(model.config.num_hidden_layers).to('cuda').eval();writer.load_state_dict(payload['writer']);writer.requires_grad_(False)
        def plain(q):return closed_think_inputs(tok,q,'cuda')['input_ids']
        with torch.no_grad():
            header=plain('')[:,:2];cache=model(input_ids=header,use_cache=True).past_key_values.to_legacy_cache()
            header_cache=tuple((k.detach(),v.detach()) for k,v in cache)
            def state(fact):
                ids=tok(fact+'\n',return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')
                return tuple((k.detach().float(),v.detach().float()) for k,v in model(input_ids=ids,use_cache=True).past_key_values.to_legacy_cache())
            def gen(q,source):
                ids=plain(q);suffix=ids[:,2:];generated=[]
                if source is not None:
                    mem=writer(source);pairs=tuple((torch.cat((hk,mem[i][0].to(hk.dtype)),2),torch.cat((hv,mem[i][1].to(hv.dtype)),2)) for i,(hk,hv) in enumerate(header_cache));p=pairs[0][0].shape[2]
                for _ in range(24):
                    added=torch.tensor([generated],device='cuda',dtype=torch.long) if generated else None
                    current=torch.cat((suffix,added),1) if added is not None else suffix
                    if source is None:out=model(input_ids=torch.cat((header,current),1),use_cache=False)
                    else:out=model(input_ids=current,past_key_values=DynamicCache.from_legacy_cache(pairs),attention_mask=torch.ones(1,p+current.shape[1],device='cuda',dtype=torch.long),cache_position=torch.arange(p,p+current.shape[1],device='cuda'),use_cache=False)
                    t=out.logits[:,-1].argmax(-1).item();generated.append(t)
                    if t==tok.eos_token_id:break
                return {'text':tok.decode(generated,skip_special_tokens=True).strip(),'tokens':generated,'ended':bool(generated and generated[-1]==tok.eos_token_id)}
            reload_rows=[]
            for split in ('train','held_name','held_value','held_relation'):
                row=next(r for r in report['rows'] if r.get('phase')=='after' and r.get('split')==split and r.get('mode')=='correct')
                name,rel,value=row['case'];subject=name+('的职业' if rel=='职业' else '最喜欢的'+rel)
                result=gen(row['question'],state(subject+'是'+value+'。'))
                reload_rows.append({'split':split,'match_original':result['tokens']==row['tokens'],**result})
            sources=[state(f) for f in facts];rows=[]
            for i,(question,target) in enumerate(zip(questions,values)):
                other=(i+1)%len(facts)
                if not question.startswith('对话中'):raise ValueError('Unexpected source question prefix')
                for style,wording in [('conversation',question),('direct',question[len('对话中'):])]:
                    query=wording+'请简洁回答。'
                    for mode,q,source in [('off',query,None),('correct',query,sources[i]),('wrong_relation',query,sources[other]),('fact_prompt',facts[i]+'\n'+query,None)]:
                        result=gen(q,source);rows.append({'index':i,'style':style,'mode':mode,'target_in_text':target in result['text'],
                            'wrong_value_in_text':values[other] in result['text'],'question':q,'target':target,**result})
                save('progress.json',{'reload':reload_rows,'rows':rows});print(json.dumps({'case':i+1,'rows':len(rows)}),flush=True)
        after={k:sha(v) for k,v in paths.items()}
        save('result.json',{'reload':reload_rows,'rows':rows,'input_hashes':hashes,'input_hashes_after':after,
            'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Previously examined dialogue values and manually formatted facts, not an independent benchmark.',
                           'No autonomous extraction, state selection, persistence, or normal-chat safety proof.',
                           'String containment is only screening; material and outputs remain local.']})
        print(json.dumps({'complete':True,'reload_match':sum(x['match_original'] for x in reload_rows),'unchanged':hashes==after,'seconds':time.perf_counter()-start}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
