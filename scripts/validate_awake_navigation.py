"""Closed-loop synthetic actions: environment checks edges, never picks moves."""
import copy,json,random,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from spora.growing import load
from scripts.train_awake_core import ROOT,distances,sha


def main():
    out=ROOT/'runs/p24-awake/navigation.json'
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1)
    model,_,_=load(ROOT/'runs/p24-awake/individual-awake.pt')
    rng=random.Random(241108);rows=[]
    for case in range(100):
        graph=torch.zeros(16,16);chain=rng.sample(range(16),16)
        for a,b in zip(chain,chain[1:]):graph[a,b]=1
        for a in range(16):
            for b in range(16):
                if a!=b and rng.random()<.04:graph[a,b]=1
        start,target=chain[0],chain[-1]
        oracle=distances(graph,target)[start] # Evaluation metric only.
        c=copy.deepcopy(model.awake_core);c.set_mode('focus')
        c.observe(graph,start,'synthetic');c.set_goal(target)
        position=start;path=[position];ticks=0;invalid=False
        for _ in range(16):
            for _ in range(40):
                s=c.tick(50_000_000);ticks+=1
                if s['outcome']!='planning':break
            if s['outcome']=='at_goal':break
            action=s['proposed_node']
            if action is None or graph[position,action]!=1:invalid=True;break
            # Environment executes and reports only the model's proposed move.
            position=action;path.append(position);c.observe(graph,position,'synthetic')
        rows.append(dict(case=case,goal_commands=1,reached=position==target,invalid_action=invalid,
                         moves=len(path)-1,oracle_moves=oracle,path=path,ticks=ticks))
    result=dict(cases=100,reached=sum(r['reached'] for r in rows),invalid_actions=sum(r['invalid_action'] for r in rows),
        shortest_paths=sum(r['reached'] and r['moves']==r['oracle_moves'] for r in rows),rows=rows,
        scope='One explicit numeric goal per episode; subsequent inputs are synthetic action-result observations, not user prompts. Environment never searches for or suggests moves. Does not establish autonomous goal invention or physical computer control.',
        source_sha256=sha(Path(__file__)))
    result['passed']=result['reached']==100 and result['invalid_actions']==0 and result['shortest_paths']==100
    out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
