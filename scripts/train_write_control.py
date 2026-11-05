"""Synthetic declared-evidence curriculum; teacher policy is not runtime code."""
import argparse,json,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.internal_memory import WriteControl,ACTIONS


def sample(generator,batch):
    source=torch.randint(4,(batch,),generator=generator)
    q=torch.rand(batch,generator=generator)
    known=torch.randint(2,(batch,),generator=generator).bool()
    same=torch.randint(2,(batch,),generator=generator).bool() & known
    recorded=(torch.rand(batch,generator=generator)>.15) & known
    previous=torch.where(recorded,torch.rand(batch,generator=generator),known.float())
    freshness=torch.randint(3,(batch,),generator=generator)-1
    freshness=torch.where(recorded,freshness,torch.ones_like(freshness))
    features=torch.cat([F.one_hot(source,4).float(),torch.stack([q,known.float(),same.float(),previous,freshness.float(),recorded.float()],-1)],-1)
    allowed=(source<2)&(q>=.7)&(freshness>=0)&((q+.05>=previous)|~known)
    allowed=torch.where(known&~recorded,(source==0)&(q>=.95),allowed)
    labels=torch.full((batch,),4)
    labels[allowed&~known]=0
    labels[allowed&known&~same]=1
    labels[allowed&known&same]=3
    labels[allowed&known&same&recorded&(q>=previous+.1)]=2
    return features,labels


@torch.no_grad()
def evaluate(model,seed,count=65536):
    features,labels=sample(torch.Generator().manual_seed(seed),count)
    probabilities=model(features).softmax(-1)
    confidence,pred=probabilities.max(-1)
    selected=pred.clone()
    selected[confidence<model.confidence]=4
    selected[features[:,2:4].sum(-1)>.5]=4
    accuracy=lambda x:{name:float((x[labels==i]==i).float().mean()) for i,name in enumerate(ACTIONS)}
    return {'cases':count,'raw_accuracy':accuracy(pred),'guarded_accuracy':accuracy(selected),
            'uncertain_fraction':float((confidence<model.confidence).float().mean()),
            'nonfactual_writes':int(((selected<2)&(features[:,2:4].sum(-1)>.5)).sum())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--balanced',action='store_true');a=p.parse_args()
    output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(a.seed)
    plan=json.loads((ROOT/'configs/p12-p13-plan.json').read_text());cfg=plan['write_training']
    model=WriteControl(cfg['hidden'],plan['write_confidence_min'])
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    gen=torch.Generator().manual_seed(plan['training_seed']+a.seed)
    start=time.perf_counter()
    revision=None
    if a.balanced:
        revision=json.loads((ROOT/'configs/p12-development-v2.json').read_text())
        pool,targets=sample(gen,revision['training_pool'])
        groups=[torch.where(targets==i)[0] for i in range(5)]
        cfg=dict(cfg,steps=revision['training_steps'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg['steps'],eta_min=.0001)
    # Counter class imbalance; weights fixed from a separate training draw.
    _,labels=sample(torch.Generator().manual_seed(plan['training_seed']+100),65536)
    weights=(labels.bincount(minlength=5).float()+1).reciprocal();weights=weights/weights.mean()
    for step in range(cfg['steps']):
        if a.balanced:
            indices=torch.cat([group[torch.randint(len(group),(cfg['batch']//5,),generator=gen)] for group in groups])
            features,target=pool[indices],targets[indices]
        else:features,target=sample(gen,cfg['batch'])
        loss=F.cross_entropy(model(features),target,weight=None if a.balanced else weights)
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if a.balanced:scheduler.step()
        if (step+1)%400==0:print(json.dumps({'step':step+1,'loss':float(loss.detach())}),flush=True)
    model.eval()
    report={'seed':a.seed,'steps':cfg['steps'],'examples':cfg['steps']*(cfg['batch']//5*5 if a.balanced else cfg['batch']),
            'development_revision':revision,
            'seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters()),
            'development':evaluate(model,plan['development_seed']), 'language_tokens':0}
    torch.save({'format':'spora-write-control-v1','config':model.config_dict(),'state':model.state_dict(),'training':report},output/'model.pt')
    (output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
