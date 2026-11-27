"""Frozen selector diagnostic: question representations and subject interventions."""
import json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha
from scripts.deepseek_dialogue_distill_20261127 import REL,statement,question
from scripts.deepseek_query_evidence_probe_20261127 import Selector

RUN=ROOT/'runs/research/selector-rotating-person-20261127-v1'
OUT=ROOT/'runs/research/query-representation-diagnostic-20261127-v1'

def main():
    OUT.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    try:
        paths={'base':MODEL/'model.safetensors','candidate':RUN/'candidate.pt','report':RUN/'result.json'}
        hashes={k:sha(p) for k,p in paths.items()}
        if hashes['base']!=WEIGHT_SHA or hashes['candidate']!=json.loads(paths['report'].read_text(encoding='utf8'))['candidate_sha256']:
            raise ValueError('input mismatch')
        payload=torch.load(paths['candidate'],map_location='cpu',weights_only=True)
        if payload['base_sha256']!=WEIGHT_SHA:raise ValueError('base mismatch')
        torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        selector=Selector(model.config.hidden_size).to('cuda').eval().requires_grad_(False);selector.load_state_dict(payload['selector'])
        people=('小陈','小吴','小邵','小黎','小姜','小宋');relations=list(REL);queries={};projected={}
        with torch.no_grad():
            for person in people:
                for relation in relations:
                    ids=closed_think_inputs(tok,question(person,relation),'cuda')['input_ids']
                    q=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,-1].float()
                    queries[(person,relation)]=q
                    projected[(person,relation)]=selector.query(F.layer_norm(q,(q.shape[-1],)))
            distances=[]
            for i,person in enumerate(people):
                for other in people[i+1:]:
                    for relation in relations:
                        a,b=(person,relation),(other,relation)
                        distances.append({'kind':'person','raw':float(1-F.cosine_similarity(queries[a],queries[b],dim=0)),
                            'projected':float(1-F.cosine_similarity(projected[a],projected[b],dim=0))})
            for person in people:
                for i,relation in enumerate(relations):
                    for other in relations[i+1:]:
                        a,b=(person,relation),(person,other)
                        distances.append({'kind':'relation','raw':float(1-F.cosine_similarity(queries[a],queries[b],dim=0)),
                            'projected':float(1-F.cosine_similarity(projected[a],projected[b],dim=0))})
            rows=[]
            for idx,r1 in enumerate(relations):
                r2=relations[(idx+3)%len(relations)];p1,p2='小黎','小姜';absent='小宋'
                facts=[(p1,r1,REL[r1][0]),(p2,r2,REL[r2][2]),(p2,r1,REL[r1][1]),(p1,r2,REL[r2][2])]
                text='小安：我们一起整理上周活动记录。\n'
                for fact in facts:text+='小安：刚才提到，'+statement(*fact)+'接下来我们再看其他安排。\n'
                text+='小安：这些资料整理完后，大家可以去休息。\n'
                ids=tok(text,add_special_tokens=False,return_tensors='pt').input_ids.to('cuda')
                hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state[0]
                for person in (p1,p2,absent):
                    score=selector(hidden,queries[(person,r1)]);rank=torch.topk(score[:-1],2).indices.tolist()
                    rows.append({'document':idx,'person_role':'first' if person==p1 else 'second' if person==p2 else 'absent',
                        'top2':rank,'null':bool(score[-1]>score[:-1].max()),
                        'null_margin':float(score[-1]-score[:-1].max())})
                print(json.dumps({'document':idx+1}),flush=True)
        after={k:sha(p) for k,p in paths.items()}
        result={'distances':distances,'rows':rows,'input_hashes':hashes,'input_hashes_after':after,
                'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'limitations':['Same known synthetic templates and candidate, not independent evaluation.',
                  'Cosine distance and changed rankings do not measure correct binding or causal necessity.',
                  'Question-only null score is compared with source-dependent maximum token score.']}
        (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        print(json.dumps({'complete':True,'unchanged':hashes==after,'seconds':time.perf_counter()-start}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
