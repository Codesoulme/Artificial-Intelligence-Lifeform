"""Train selector on synthetic multi-person evidence binding and missing entities."""
import json,random,sys,time,traceback
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

PARENT=ROOT/'runs/research/query-evidence-probe-20261127-v1'
OUT=ROOT/'runs/research/selector-multifact-train-20261127-v2'
SEED,STEPS,LR=112746,576,0.00015

def main():
    OUT.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    def save(name,data):
        p=OUT/(name+'.tmp');p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');p.replace(OUT/name)
    try:
        paths={'base':MODEL/'model.safetensors','parent':PARENT/'candidate.pt','report':PARENT/'result.json'}
        hashes={k:sha(p) for k,p in paths.items()};report=json.loads(paths['report'].read_text(encoding='utf8'))
        if hashes['base']!=WEIGHT_SHA or hashes['parent']!=report['candidate_sha256']:raise ValueError('unexpected inputs')
        payload=torch.load(paths['parent'],map_location='cpu',weights_only=True)
        if payload['base_sha256']!=WEIGHT_SHA:raise ValueError('wrong base')
        random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        selector=Selector(model.config.hidden_size).to('cuda');selector.load_state_dict(payload['selector'])
        rels=list(REL);people=('小陈','小吴');absent='小邵';docs=[];queries={};samples=[]
        with torch.no_grad():
            for p in (*people,absent):
                for r in rels:
                    ids=closed_think_inputs(tok,question(p,r),'cuda')['input_ids']
                    queries[(p,r)]=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,-1].detach().cpu()
            for i,r1 in enumerate(rels):
                r2=rels[(i+3)%len(rels)];missing=rels[(i+5)%len(rels)]
                for variant in range(2):
                    facts=[(people[0],r1,REL[r1][0]),(people[1],r2,REL[r2][2]),
                           (people[1],r1,REL[r1][1]),(people[0],r2,REL[r2][2])]
                    if variant:facts=list(reversed(facts))
                    text='小安：我们核对一下大家提供的资料。\n';spans={}
                    for fact in facts:
                        sentence=statement(*fact);prefix='小安：现在记下，';pos=len(text)+len(prefix)+sentence.index(fact[2])
                        spans[(fact[0],fact[1])]=(pos,pos+len(fact[2]))
                        text+=prefix+sentence+'然后我们再商量别的安排。\n'
                    text+='小安：核对完毕，谢谢大家。\n'
                    item=tok(text,add_special_tokens=False,return_offsets_mapping=True,return_tensors='pt')
                    hidden=model.model(input_ids=item['input_ids'].to('cuda'),use_cache=False).last_hidden_state[0].detach().cpu()
                    offs=item['offset_mapping'][0].tolist();targets={}
                    for key,(a,b) in spans.items():
                        targets[key]=[j for j,(x,y) in enumerate(offs) if x<b and y>a]
                        if not targets[key]:raise ValueError('empty target')
                    doc=len(docs);docs.append((hidden,targets))
                    for p in people:
                        for r in (r1,r2):samples.append({'doc':doc,'query':(p,r),'positive':True})
                    samples.extend([{'doc':doc,'query':(people[0],missing),'positive':False},
                                    {'doc':doc,'query':(absent,r1),'positive':False}])
                print(json.dumps({'relations_done':i+1,'docs':len(docs)}),flush=True)
        del model;torch.cuda.empty_cache()
        opt=torch.optim.AdamW(selector.parameters(),lr=LR,weight_decay=0);rng=random.Random(SEED);order=samples.copy();trace=[]
        for step in range(STEPS):
            if step%len(order)==0:rng.shuffle(order)
            row=order[step%len(order)];hidden,targets=docs[row['doc']];scores=selector(hidden.to('cuda'),queries[row['query']].to('cuda'))
            logp=F.log_softmax(scores,dim=0);loss=-logp[targets[row['query']]].mean() if row['positive'] else -logp[-1]
            opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(selector.parameters(),1,error_if_nonfinite=True));opt.step()
            trace.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':norm})
            if step%96==0 or step==STEPS-1:save('training-progress.json',{'trace':trace});print(json.dumps(trace[-1]),flush=True)
        selector.eval();rows=[]
        with torch.no_grad():
            for row in samples:
                hidden,targets=docs[row['doc']];scores=selector(hidden.to('cuda'),queries[row['query']].to('cuda'))
                rank=torch.topk(scores[:-1],2).indices.tolist();null=bool((scores[-1]>scores[:-1].max()).item())
                target=targets.get(row['query'],[])
                rows.append({'positive':row['positive'],'null':null,'top2_any':bool(set(rank)&set(target)),
                    'top2_all':bool(target) and set(target)<=set(rank),'target_tokens':len(target)})
        torch.save({'selector':{k:v.detach().cpu() for k,v in selector.state_dict().items()},'base_sha256':WEIGHT_SHA,
                    'parent_sha256':hashes['parent'],'seed':SEED,'dim':payload['dim']},OUT/'candidate.pt')
        after={k:sha(p) for k,p in paths.items()}
        save('result.json',{'rows':rows,'trace':trace,'steps':STEPS,'lr':LR,'seed':SEED,
            'input_hashes':hashes,'input_hashes_after':after,'candidate_sha256':sha(OUT/'candidate.pt'),
            'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Synthetic multi-fact template; no independent source, free answers, or persistence.',
                'Answer locations used as training supervision; fixed two-token selection may miss longer values.']})
        print(json.dumps({'complete':True,'unchanged':hashes==after,'seconds':time.perf_counter()-start}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
