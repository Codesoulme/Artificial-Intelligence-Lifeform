"""Fixed-model protection ablations: replay and soft protected-subspace delta."""
import argparse
import json
from pathlib import Path
import time
import torch
from .checkpoint import load_checkpoint
from .p2 import stream,write,Event


def protected_delta(model,fast,velocity,event,keys_to_preserve,ridge=.05):
    """Soft projector; target key excluded. No persistent answer table."""
    output=[]
    key_embed=model.embedding(event.key+model.config.key_start)
    value_embed=model.embedding(event.value+model.config.value_start)
    for site,params in zip(model.memories,fast):
        q=site.address(key_embed)
        target=site.target(value_embed).detach()
        protected=site.address(model.embedding(keys_to_preserve+model.config.key_start)).detach()
        covariance=protected.transpose(1,2)@protected
        identity=torch.eye(q.shape[-1],device=q.device).unsqueeze(0)
        direction=torch.linalg.solve(covariance+ridge*identity,q.unsqueeze(-1)).squeeze(-1)
        direction=direction/(direction*q).sum(-1,keepdim=True).clamp_min(1e-5)
        residual=target-site.retrieve(params,q)
        change=torch.sigmoid(site.log_lr)*residual.unsqueeze(-1)*direction.unsqueeze(1)
        # Explicit bounded update: uncertain large projection is not silently accepted.
        norm=change.flatten(1).norm(dim=-1).clamp_min(1e-8)
        scale=(20./norm).clamp(max=1)
        updated=params[0]+event.trusted[:,None,None]*change*scale[:,None,None]
        output.append([updated.detach().requires_grad_(True)])
    return output,velocity


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--worlds',type=int,default=32)
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists():raise FileExistsError('Preserve prior ablation')
    torch.set_num_threads(4)
    model,_,_,_=load_checkpoint(args.checkpoint,'cpu')
    methods=['none','replay']+(['soft_projection'] if model.config.memory_kind=='delta' else [])
    report=[]
    for method in methods:
        for pairs in (16,64):
            generator=torch.Generator().manual_seed(530000+pairs)
            totals={'correction':0,'unrelated':0,'final':0,'updates':0};started=time.perf_counter()
            for offset in range(0,args.worlds,8):
                batch=min(8,args.worlds-offset)
                events,keys,original,current=stream(generator,batch,pairs,model.config,'cpu')
                fast,velocity=model.initial_state(batch)
                knowledge=original.clone()
                for event in events:
                    if method=='soft_projection' and event.kind=='correction':
                        mask=keys!=event.key[:,None]
                        protected=keys[mask].reshape(batch,pairs-1)
                        fast,velocity=protected_delta(model,fast,velocity,event,protected)
                        totals['updates']+=batch
                    else:
                        fast,velocity,steps=write(model,fast,velocity,event,'one')
                        totals['updates']+=float(steps.sum())
                    if event.kind=='correction':
                        positions=(keys==event.key[:,None]).to(torch.int64).argmax(1)
                        knowledge[torch.arange(batch),positions]=event.value
                        if method=='replay':
                            # One rehearsal of every other fact, values from actual past observations only.
                            for pos in range(pairs):
                                gate=(keys[:,pos]!=event.key).float()
                                rehearsal=Event(keys[:,pos],knowledge[:,pos],gate,'replay')
                                fast,velocity,steps=write(model,fast,velocity,rehearsal,'one')
                                totals['updates']+=float(steps.sum())
                        with torch.no_grad():pred=model(event.key,fast).argmax(-1)
                        totals['correction']+=int((pred==event.value).sum())
                with torch.no_grad():pred=model.predict_many(keys,fast).argmax(-1)
                totals['unrelated']+=int((pred[:,4:]==current[:,4:]).sum())
                totals['final']+=int((pred==current).sum())
            row={'method':method,'pairs':pairs,'correction':totals['correction']/(args.worlds*4),
                 'unrelated':totals['unrelated']/(args.worlds*(pairs-4)),
                 'final':totals['final']/(args.worlds*pairs),'inner_updates':totals['updates'],
                 'seconds':time.perf_counter()-started}
            report.append(row);print(json.dumps(row),flush=True)
    result={'checkpoint':args.checkpoint,'worlds':args.worlds,'results':report,
            'scope':'Development-only ablation; no tuning or checkpoint writes',
            'caveats':['Replay explicitly consumes historical training samples and extra compute',
                       'Soft projection is a small linear-memory adaptation, not AlphaEdit replication',
                       'Projection uses previously observed keys during learning only; not a query answer database']}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
