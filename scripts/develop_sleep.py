"""Fresh development stream; choose a bottleneck before final audits."""
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.checkpoint import load_checkpoint
from spora.growing import new_state,observe,grow,query,scalars
from spora.sleep_memory import propose,behavior_gate


def main():
    output=ROOT/'runs/p10-development'
    if output.exists():raise FileExistsError(output)
    output.mkdir();torch.set_num_threads(2)
    model,_,_,_=load_checkpoint(ROOT/'checkpoints/p5b-orthogonal90-seed7.pt','cuda')
    generator=torch.Generator().manual_seed(2210000)
    keys=torch.randint(64,(64,),generator=generator)
    values=torch.randint(16,(64,),generator=generator)
    trust=(torch.rand(64,generator=generator)>.25).long()
    state=new_state(model)
    for key,value,trusted in zip(keys,values,trust):
        state=observe(model,state,int(key),int(value),int(trusted))
    results=[]
    for rank in (16,24,32):
        for method in ('svd','distill'):
            candidate,training=propose(model,state,0,rank=rank,steps=600,seed=2210000,method=method)
            gate=behavior_gate(model,state,candidate,0)
            with torch.no_grad():
                k=keys.cuda();t=torch.arange(64,device='cuda');m=torch.ones_like(k)
                prediction=query(model,candidate,k,m,t).argmax(-1)
                history=float((prediction==values.cuda()).float().mean())
                untrusted=float((prediction[trust==0]==values.cuda()[trust==0]).float().mean())
            row={'rank':rank,'method':method,'training':training,'gate':gate,'history':history,'untrusted_history':untrusted}
            results.append(row);print(json.dumps(row),flush=True)
            (output/'results.json').write_text(json.dumps(results,indent=2))
    passing=[r for r in results if r['method']=='distill' and r['gate']['passed'] and min(r['history'],r['untrusted_history'])>=.95 and r['training']['state_reduction']>=.2]
    if not passing:raise RuntimeError('Development has no passing trained transfer')
    chosen=min(passing,key=lambda r:r['rank'])
    selection={'rank':chosen['rank'],'steps':600,'method':'distill','seed':2210000,'development_only':True,'threshold':.95}
    (ROOT/'configs/p10-selection.json').write_text(json.dumps(selection,indent=2))
    print(json.dumps({'selected':selection}),flush=True)

if __name__=='__main__':main()
