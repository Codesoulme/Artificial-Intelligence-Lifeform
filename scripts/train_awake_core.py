"""Train and independently evaluate P24 finite recurrent planning from scratch."""
import argparse,copy,json,random,sys,time,hashlib
from collections import deque
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from spora.awake import AwakeCore,attach_awake
from spora.growing import load,save,query
from spora.state_codec import flatten
from spora.durable import TensorStore

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'checkpoints/individual-prechinese-v3.pt'
PARENT_SHA='6269d7749148c95ed4653928f61642c8d93098defed649baab3f807c71494db9'

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def distances(graph,target):
    # Evaluation oracle only; never passed to the model.
    d={target:0};q=deque([target])
    while q:
        v=q.popleft()
        for u in range(len(graph)):
            if graph[u,v] and u not in d:d[u]=d[v]+1;q.append(u)
    return d

def evaluate(trained,seed,cases=200):
    rng=random.Random(seed);rows=[]
    for i in range(cases):
        n=trained.config.nodes;graph=torch.zeros(n,n)
        for u in range(n):
            for v in range(n):
                if u!=v and rng.random()<.10:graph[u,v]=1
        start,target=rng.sample(range(n),2)
        if i%2==0:
            path=rng.sample([j for j in range(n) if j not in (start,target)],rng.randrange(2,n-1))
            chain=[start]+path+[target]
            for u,v in zip(chain,chain[1:]):graph[u,v]=1
        d=distances(graph,target)
        core=copy.deepcopy(trained);core.observe(graph,start,'synthetic');core.set_goal(target)
        core.set_mode('focus' if i%2 else 'relax')
        for ticks in range(1,2*n+5):
            status=core.tick(int(core.interval()*1e9))
            if status['outcome'] not in ('planning','idle'):break
        action=status['proposed_node']
        correct=(status['outcome']=='unreachable_estimate') if start not in d else (
            status['outcome']=='proposal' and action in d and graph[start,action]==1 and d[action]==d[start]-1)
        rows.append(dict(case=i,reachable=start in d,distance=d.get(start),correct=bool(correct),
                         ticks=ticks,neural_steps=status['neural_steps'],outcome=status['outcome']))
    return dict(seed=seed,cases=cases,accuracy=sum(r['correct'] for r in rows)/cases,
                reachable=sum(r['reachable'] for r in rows),rows=rows)

def train(seed,steps):
    torch.manual_seed(seed);core=AwakeCore();opt=torch.optim.Adam(core.parameters(),lr=.003)
    start=time.perf_counter()
    for step in range(steps):
        goal=torch.randint(0,2,(256,1)).float();neighbor=torch.rand(256,1)
        neighbor[:64]=0
        target=torch.maximum(goal,.85*neighbor)
        raw=core.transition(torch.cat((goal,neighbor),1))
        # Learn a slightly negative pre-clamp floor to avoid spurious reachability.
        target=torch.where((goal==0)&(neighbor==0),-.002,target)
        flags=torch.randint(0,2,(256,3)).float()
        budget=torch.where(flags[:,1]==0,0,torch.where(flags[:,2]>0,2,flags[:,0].long()))
        loss=F.mse_loss(raw,target)*10+F.cross_entropy(core.budget(flags),budget)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if (step+1)%500==0:print(json.dumps(dict(seed=seed,step=step+1,loss=float(loss.detach()))),flush=True)
    return core.eval(),dict(seed=seed,steps=steps,loss=float(loss.detach()),seconds=time.perf_counter()-start)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='runs/p24-awake');p.add_argument('--steps',type=int,default=3000)
    args=p.parse_args();out=ROOT/args.output;out.mkdir(exist_ok=False)
    if sha(PARENT)!=PARENT_SHA:raise ValueError('Parent changed')
    torch.set_num_threads(2);results=[]
    for seed in (7,11,23):
        core,training=train(seed,args.steps)
        dev=evaluate(core,seed+90000,200)
        results.append(dict(training=training,development=dev))
        torch.save(dict(config=core.config_dict(),state=core.state_dict(),seed=seed),out/f'core-seed{seed}.pt')
        print(json.dumps(dict(seed=seed,accuracy=dev['accuracy'])),flush=True)
    passed=all(r['development']['accuracy']>=.98 for r in results)
    report=dict(status='finite_development_pass' if passed else 'development_failed',results=results,
        source_sha256={name:sha(ROOT/name) for name in ('spora/awake.py','spora/awake_session.py','scripts/train_awake_core.py')},
        scope='Neural local value propagation learned on synthetic scalar Bellman targets; held random graphs evaluate multi-step shortest-path suggestions. Not language, general reasoning, or consciousness. Learned budget imitation, not discovered optimal resource allocation.')
    if passed:
        payload=torch.load(out/'core-seed7.pt',weights_only=True);core=AwakeCore();core.load_state_dict(payload['state'])
        model,state,_=load(PARENT);_,before=flatten(model,state)
        before={k:v.detach().clone() for k,v in before.items()}
        attach_awake(model,core)
        _,after=flatten(model,state)
        exact=all(torch.equal(v,after[k]) for k,v in before.items())
        if not exact:raise AssertionError('Prior individual changed')
        metadata=dict(stage='P24 awake candidate',parent_sha256=PARENT_SHA,
                      scope='Finite numeric graph input; no natural-language semantic alignment; sleep pauses planning, full dream learning pending.')
        save(out/'individual-awake.pt',model,state,metadata)
        with TensorStore.create(out/'individual-awake.sqlite',model,state,metadata) as store:pass
        report.update(prior_tensors_exact=exact,parameters=sum(p.numel() for p in core.parameters()),
                      working_tensor_bytes=sum(t.numel()*t.element_size() for t in core.buffers()),
                      individual_sha256=sha(out/'individual-awake.pt'))
    report['original_file_unchanged']=sha(PARENT)==PARENT_SHA
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)

if __name__=='__main__':main()
