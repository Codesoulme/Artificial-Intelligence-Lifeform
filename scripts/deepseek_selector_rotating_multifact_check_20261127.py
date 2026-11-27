"""Frozen selector stress test with competing synthetic people and facts."""
import json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha
from scripts.deepseek_dialogue_distill_20261127 import REL,statement,question
from scripts.deepseek_query_evidence_probe_20261127 import Selector

RUN=ROOT/'runs/research/selector-rotating-person-20261127-v1'
OUT=ROOT/'runs/research/selector-rotating-multifact-check-20261127-v1'

def main():
    OUT.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    def save(name,data):
        p=OUT/(name+'.tmp');p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');p.replace(OUT/name)
    try:
        paths={'base':MODEL/'model.safetensors','selector':RUN/'candidate.pt','report':RUN/'result.json'}
        hashes={k:sha(p) for k,p in paths.items()};report=json.loads(paths['report'].read_text(encoding='utf8'))
        if hashes['base']!=WEIGHT_SHA or hashes['selector']!=report['candidate_sha256']:raise ValueError('unexpected inputs')
        payload=torch.load(paths['selector'],map_location='cpu',weights_only=True)
        if payload['base_sha256']!=WEIGHT_SHA:raise ValueError('wrong base')
        torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        selector=Selector(model.config.hidden_size).to('cuda').eval().requires_grad_(False);selector.load_state_dict(payload['selector'])
        relations=list(REL);people=('小黎','小姜');unknown='小宋';rows=[]
        with torch.no_grad():
            for index,r1 in enumerate(relations):
                r2=relations[(index+3)%len(relations)];missing=relations[(index+5)%len(relations)]
                assert len({r1,r2,missing})==3
                facts=[(people[0],r1,REL[r1][0]),(people[1],r2,REL[r2][2]),
                       (people[1],r1,REL[r1][1]),(people[0],r2,REL[r2][2])]
                text='小安：我们一起整理上周活动记录。\n'
                spans={}
                for case in facts:
                    sentence=statement(*case)
                    if sentence in text:raise ValueError('duplicate fact')
                    prefix='小安：刚才提到，';start_sentence=len(text)+len(prefix)
                    value_start=start_sentence+sentence.index(case[2])
                    spans[(case[0],case[1])]=(value_start,value_start+len(case[2]))
                    text+=prefix+sentence+'接下来我们再看其他安排。\n'
                text+='小安：这些资料整理完后，大家可以去休息。\n'
                encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True,return_tensors='pt')
                hidden=model.model(input_ids=encoded['input_ids'].to('cuda'),use_cache=False).last_hidden_state[0]
                offsets=encoded['offset_mapping'][0].tolist()
                def label(k):
                    a,b=spans[k];return [i for i,(x,y) in enumerate(offsets) if x<b and y>a]
                targets=[(p,r) for p in people for r in (r1,r2)]
                for p,r in targets+[(people[0],missing),(unknown,r1)]:
                    is_pos=(p,r) in spans
                    ids=closed_think_inputs(tok,question(p,r),'cuda')['input_ids']
                    q=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,-1]
                    scores=selector(hidden,q);rank=torch.topk(scores[:-1],2).indices.tolist()
                    null=bool((scores[-1]>scores[:-1].max()).item())
                    goal=label((p,r)) if is_pos else []
                    counterpart=label((people[1] if p==people[0] else people[0],r)) if is_pos else []
                    if is_pos and (not goal or not counterpart):raise ValueError('missing target span')
                    rows.append({'document':index,'subject':p,'relation':r,
                        'category':'shared_value' if is_pos and r==r2 else 'different_value' if is_pos else 'missing_relation' if p!=unknown else 'missing_person',
                        'positive':is_pos,'null':null,'target_tokens':len(goal),'top2_any':bool(set(rank)&set(goal)),
                        'top2_all':bool(goal) and set(goal)<=set(rank),'counterpart_any':bool(set(rank)&set(counterpart)),
                        'top2':rank,'null_margin':float(scores[-1]-scores[:-1].max())})
                save('progress.json',{'rows':rows});print(json.dumps({'document':index+1,'rows':len(rows)}),flush=True)
        after={k:sha(p) for k,p in paths.items()}
        save('result.json',{'rows':rows,'input_hashes':hashes,'input_hashes_after':after,
            'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Synthetic multi-fact data; relations and some values were seen in training.',
                'Ground-truth position used for scoring only; no model answer generation or memory persistence.',
                'Shared-value cases require exact occurrence, a stricter criterion than answer text alone.']})
        print(json.dumps({'complete':True,'unchanged':hashes==after,'seconds':time.perf_counter()-start}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
