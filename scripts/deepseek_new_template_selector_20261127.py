"""Frozen selectors on new people, context wording, and fact order."""
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
from scripts.deepseek_token_interaction_selector_20261127 import TokenInteractionSelector

RUNS={'single':ROOT/'runs/research/selector-rotating-person-20261127-v1',
      'token':ROOT/'runs/research/selector-token-interaction-20261127-v1'}
OUT=ROOT/'runs/research/new-template-selector-20261127-v1'

def main():
    OUT.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    def save(name,data):
        p=OUT/(name+'.tmp');p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');p.replace(OUT/name)
    try:
        paths={'base':MODEL/'model.safetensors'}
        for n,p in RUNS.items():paths[n]=p/'candidate.pt';paths[n+'_report']=p/'result.json'
        hashes={k:sha(p) for k,p in paths.items()}
        if hashes['base']!=WEIGHT_SHA:raise ValueError('wrong base')
        for n,p in RUNS.items():
            if hashes[n]!=json.loads((p/'result.json').read_text(encoding='utf8'))['candidate_sha256']:raise ValueError('wrong candidate')
        torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        selectors={}
        for label,cls in [('single',Selector),('token',TokenInteractionSelector)]:
            payload=torch.load(paths[label],map_location='cpu',weights_only=True)
            if payload['base_sha256']!=WEIGHT_SHA:raise ValueError('base mismatch')
            w=cls(model.config.hidden_size).to('cuda').eval().requires_grad_(False);w.load_state_dict(payload['selector']);selectors[label]=w
        people=('小夏','小何');absent='小唐';rels=list(REL);rows=[]
        with torch.no_grad():
            qs={}
            for person in (*people,absent):
                for relation in rels:
                    ids=closed_think_inputs(tok,question(person,relation),'cuda')['input_ids']
                    qs[(person,relation)]=model.model(input_ids=ids,use_cache=False).last_hidden_state[0]
            for index,r1 in enumerate(rels):
                r2=rels[(index+3)%len(rels)];missing=rels[(index+5)%len(rels)]
                for order in range(2):
                    facts=[(people[0],r1,REL[r1][0]),(people[1],r2,REL[r2][2]),
                           (people[1],r1,REL[r1][1]),(people[0],r2,REL[r2][2])]
                    if order:facts.reverse()
                    text='记录员：以下是大家刚才分享的情况。\n';spans={}
                    for fact in facts:
                        sentence=statement(*fact);prefix='记录员：';pos=len(text)+len(prefix)+sentence.index(fact[2])
                        spans[(fact[0],fact[1])]=(pos,pos+len(fact[2]))
                        text+=prefix+sentence+'谢谢你的分享。\n'
                    text+='记录员：今天的记录就到这里。\n'
                    item=tok(text,add_special_tokens=False,return_offsets_mapping=True,return_tensors='pt')
                    hidden=model.model(input_ids=item['input_ids'].to('cuda'),use_cache=False).last_hidden_state[0]
                    offsets=item['offset_mapping'][0].tolist()
                    def label(key):
                        a,b=spans[key];return [j for j,(x,y) in enumerate(offsets) if x<b and y>a]
                    targets=[(p,r) for p in people for r in (r1,r2)]
                    for person,relation in targets+[(people[0],missing),(absent,r1)]:
                        positive=(person,relation) in spans
                        goal=label((person,relation)) if positive else []
                        other=label((people[1] if person==people[0] else people[0],relation)) if positive else []
                        if positive and (not goal or not other):raise ValueError('empty target')
                        for name,w in selectors.items():
                            q=qs[(person,relation)] if name=='token' else qs[(person,relation)][-1]
                            score=w(hidden,q);rank=torch.topk(score[:-1],2).indices.tolist()
                            rows.append({'candidate':name,'document':index*2+order,'order':order,'positive':positive,
                                'category':'shared_value' if positive and relation==r2 else 'different_value' if positive else 'missing_person' if person==absent else 'missing_relation',
                                'null':bool(score[-1]>score[:-1].max()),'target_tokens':len(goal),
                                'top2_any':bool(set(rank)&set(goal)),'top2_all':bool(goal) and set(goal)<=set(rank),
                                'counterpart_any':bool(set(rank)&set(other)),'top2':rank})
                save('progress.json',{'rows':rows});print(json.dumps({'relation':index+1,'rows':len(rows)}),flush=True)
        after={k:sha(p) for k,p in paths.items()}
        save('result.json',{'rows':rows,'input_hashes':hashes,'input_hashes_after':after,'seconds':time.perf_counter()-start,
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
             'limitations':['New names and context wording, but fact and question templates and values overlap training.',
                 'Small synthetic dataset, ground-truth positions scored only, no generation or persistence.',
                 'Two-token cap can never fully cover longer targets.']})
        print(json.dumps({'complete':True,'unchanged':hashes==after,'seconds':time.perf_counter()-start}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
