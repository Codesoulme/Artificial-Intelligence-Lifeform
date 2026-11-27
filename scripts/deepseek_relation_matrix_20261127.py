"""Compare relation-pair generalization of two frozen research writers."""
import json, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_residual_kv_writer_20261127 import ResidualWriter
from scripts.deepseek_paired_readout_20261127 import REL, statement, question

OUT = ROOT/'runs/research/relation-matrix-20261127-v1'
RUNS = {'parent': ROOT/'runs/research/residual-kv-writer-20261127-v1',
        'coverage': ROOT/'runs/research/relation-coverage-20261127-v1'}

def main():
    OUT.mkdir(parents=True, exist_ok=False); start=time.perf_counter()
    def save(name, data):
        p=OUT/(name+'.tmp');p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');p.replace(OUT/name)
    try:
        torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
        paths={'base':MODEL/'model.safetensors'}
        for name,run in RUNS.items():
            paths[name]=run/'candidate.pt';paths[name+'_report']=run/'result.json'
        hashes={k:sha(v) for k,v in paths.items()}
        if hashes['base']!=WEIGHT_SHA:raise ValueError('Unexpected base')
        reports={name:json.loads((run/'result.json').read_text(encoding='utf8')) for name,run in RUNS.items()}
        for name in RUNS:
            if hashes[name]!=reports[name]['candidate_sha256']:raise ValueError('Unexpected candidate')
        pairs={(r['state_relation'],r['question_relation']) for r in reports['coverage']['trace']}
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        def plain(q):return closed_think_inputs(tok,q,'cuda')['input_ids']
        with torch.no_grad():
            header=plain('')[:,:2]
            header_cache=tuple((k.detach(),v.detach()) for k,v in model(input_ids=header,use_cache=True).past_key_values.to_legacy_cache())
            states={}
            for relation,values in REL.items():
                ids=tok(statement('小陈',relation,values[0])+'\n',add_special_tokens=False,return_tensors='pt').input_ids.to('cuda')
                states[relation]=tuple((k.detach().float(),v.detach().float()) for k,v in model(input_ids=ids,use_cache=True).past_key_values.to_legacy_cache())
            rows=[]
            for label in RUNS:
                payload=torch.load(paths[label],map_location='cpu',weights_only=True)
                if payload['base_sha256']!=WEIGHT_SHA:raise ValueError('Mismatched base')
                writer=ResidualWriter(model.config.num_hidden_layers).to('cuda').eval()
                writer.load_state_dict(payload['writer']);writer.requires_grad_(False)
                for state_rel in REL:
                    memory=writer(states[state_rel]);cache=tuple((torch.cat((hk,memory[i][0].to(hk.dtype)),2),torch.cat((hv,memory[i][1].to(hv.dtype)),2)) for i,(hk,hv) in enumerate(header_cache));p=cache[0][0].shape[2]
                    for query_rel in REL:
                        ids=plain(question('小陈',query_rel))
                        if not torch.equal(ids[:,:2],header):raise ValueError('header mismatch')
                        suffix=ids[:,2:];tokens=[]
                        for _ in range(24):
                            current=torch.cat((suffix,torch.tensor([tokens],device='cuda')),1) if tokens else suffix
                            logits=model(input_ids=current,past_key_values=DynamicCache.from_legacy_cache(cache),
                                attention_mask=torch.ones(1,p+current.shape[1],device='cuda',dtype=torch.long),
                                cache_position=torch.arange(p,p+current.shape[1],device='cuda'),use_cache=False).logits
                            t=logits[:,-1].argmax(-1).item();tokens.append(t)
                            if t==tok.eos_token_id:break
                        text=tok.decode(tokens,skip_special_tokens=True).strip()
                        split='answerable' if state_rel==query_rel else ('seen_pair' if (state_rel,query_rel) in pairs else 'unseen_pair')
                        rows.append({'candidate':label,'state_relation':state_rel,'query_relation':query_rel,
                            'split':split,'text':text,'tokens':tokens,'ended':tokens[-1]==tok.eos_token_id,
                            'target_in_text':REL[state_rel][0] in text if split=='answerable' else None,
                            'state_value_in_text':REL[state_rel][0] in text,'exact_reject':text=='我不知道。'})
                    save('progress.json',{'rows':rows});print(json.dumps({'candidate':label,'state':state_rel,'rows':len(rows)},ensure_ascii=True),flush=True)
                del writer
        after={k:sha(v) for k,v in paths.items()}
        save('result.json',{'rows':rows,'seen_pairs':len(pairs),'input_hashes':hashes,'input_hashes_after':after,
            'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Training-style facts, no independent held-out persons, values, or source sessions.',
                'Only one value per state and no autonomous selection or persistence.',
                'Exact refusal and substring matches are screening, not complete semantic scoring.']})
        print(json.dumps({'complete':True,'unchanged':hashes==after,'seconds':time.perf_counter()-start}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
