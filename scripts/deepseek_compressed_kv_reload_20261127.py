"""Independent reload check of research-only compressed KV slot artifacts."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.cache_utils import DynamicCache
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha

RUN=ROOT/'runs/research/compressed-kv-fit-20261127-v1'
OUT=RUN/'reload-check.json'

def main():
    if OUT.exists():raise FileExistsError(OUT)
    start=time.perf_counter();torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
    report_path=RUN/'result.json';report_sha=sha(report_path);report=json.loads(report_path.read_text(encoding='utf8'))
    weight=MODEL/'model.safetensors';before=sha(weight)
    if before!=WEIGHT_SHA or before!=report['base_sha_before']:raise ValueError('Base mismatch')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
    def plain(q):return closed_think_inputs(tok,q,'cuda')['input_ids']
    with torch.no_grad():
        header=plain('')[:,:2];cached=model(input_ids=header,use_cache=True).past_key_values.to_legacy_cache();rows=[]
        for entry in report['states']:
            path=Path(entry['path'])
            if sha(path)!=entry['sha256']:raise ValueError('State digest mismatch')
            payload=torch.load(path,map_location='cpu',weights_only=True)
            if list(payload['case'])!=entry['case'] or payload['base_sha256']!=WEIGHT_SHA:raise ValueError('Candidate identity mismatch')
            keys=[x.to('cuda') for x in payload['keys']];vals=[x.to('cuda') for x in payload['values']]
            for condition in ('train_question','held_wording'):
                original=next(r for r in report['rows'] if r.get('case')==entry['case'] and r['phase']=='after' and r['condition']==condition)
                q=original['question'];suffix=plain(q)[:,2:];added=[]
                for _ in range(24):
                    entries=tuple((torch.cat((hk,keys[i].to(hk.dtype)),2),torch.cat((hv,vals[i].to(hv.dtype)),2)) for i,(hk,hv) in enumerate(cached))
                    cache=DynamicCache.from_legacy_cache(entries);p=entries[0][0].shape[2]
                    ids=torch.cat((suffix,torch.tensor([added],device='cuda',dtype=torch.long)),1) if added else suffix
                    out=model(input_ids=ids,past_key_values=cache,attention_mask=torch.ones(1,p+ids.shape[1],device='cuda',dtype=torch.long),cache_position=torch.arange(p,p+ids.shape[1],device='cuda'),use_cache=False)
                    token=out.logits[:,-1].argmax(-1).item();added.append(token)
                    if token==tok.eos_token_id:break
                rows.append({'case':entry['case'],'condition':condition,'tokens':added,'match_original':added==original['tokens'],
                             'text':tok.decode(added,skip_special_tokens=True).strip()})
            print(json.dumps({'case':entry['case'],'rows':len(rows)},ensure_ascii=True),flush=True)
    checks={'base_unchanged':before==sha(weight),'result_unchanged':report_sha==sha(report_path),
            'states_unchanged':all(sha(Path(e['path']))==e['sha256'] for e in report['states'])}
    result={'rows':rows,'checks':checks,'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Reloads hand-selected research states only, not a formal individual or automated memory selection.',
                           'Only same prompts, deterministic decoding and same environment are compared.']}
    with OUT.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'match':sum(x['match_original'] for x in rows),'total':len(rows),'checks':checks,'seconds':result['seconds']}),flush=True)

if __name__=='__main__':main()
