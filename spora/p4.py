"""Separated temporal metrics. Environment labels never enter the model."""
import torch
from torch.nn import functional as F
from .temporal import reset_bank
from .temporal_tasks import events,queries


def predictions(model,fast,groups):
    names=('current','history','invalid','future')
    sizes=[groups[name][0].shape[1] for name in names]
    inputs=[torch.cat([groups[name][i] for name in names],1) for i in range(3)]
    logits=model.predict_many(inputs[0],fast,inputs[1],inputs[2])
    return dict(zip(names,logits.split(sizes,dim=1)))


def task_loss(model,fast,groups):
    logits=predictions(model,fast,groups)
    losses=[]
    for name in ('current','history','invalid','future'):
        per_query=F.cross_entropy(logits[name].flatten(0,1),groups[name][3].flatten(),reduction='none').reshape(groups[name][0].shape)
        if name=='future':
            mask=groups['future_mask']
            if not bool(mask.any()):continue
            per_query=per_query[mask]
        losses.append(per_query.mean())
    return torch.stack(losses).mean()


def consume(model,record,history=True,no_write=False):
    fast,velocity=model.initial_state(record['keys'].shape[0])
    if not no_write:
        for i in range(record['keys'].shape[1]):
            fast,velocity=model.write_temporal(fast,velocity,record['keys'][:,i],record['values'][:,i],
                record['times'][:,i],record['trusted'][:,i],history=history)
    return fast,velocity


def counts(model,fast,groups):
    predicted={name:x.argmax(-1) for name,x in predictions(model,fast,groups).items()}
    matches={name:predicted[name]==groups[name][3] for name in predicted}
    trust=groups['history_trust'].bool()
    parts={'current':matches['current'][:,:4],'current_unknown':matches['current'][:,4:],
           'history':matches['history'],'trusted_history':matches['history'][trust],
           'untrusted_history':matches['history'][~trust],
           'old_history':matches['history'][:,:max(1,matches['history'].shape[1]//2)],
           'invalid_pair_unknown':matches['invalid'],'future_unknown':matches['future'][groups['future_mask']]}
    width=max(1,matches['history'].shape[1]//4)
    for quarter in range(4):
        parts[f'history_quarter_{quarter+1}']=matches['history'][:,quarter*width:(quarter+1)*width]
    return {name:{'correct':int(x.sum()),'total':x.numel()} for name,x in parts.items()}


@torch.no_grad()
def evaluate(model,worlds=32,length=8,seed=1010000,control='normal',batch=4):
    was_training=model.training;model.eval()
    generator=torch.Generator().manual_seed(seed+length)
    device=next(model.parameters()).device
    total={}
    for start in range(0,worlds,batch):
        record=events(generator,min(batch,worlds-start),length,model.config,device)
        fast,_=consume(model,record,history=control!='no-history',no_write=control=='no-write')
        if control in ('reset-history','reset-current'):
            fast=reset_bank(model,fast,1 if control=='reset-history' else 0)
        groups=queries(record,length,model.config)
        if control=='shuffled-time':
            k,m,t,y=groups['history']
            # Original labels with rotated timestamps intentionally break event association.
            groups['history']=(k,m,t.roll(1,1),y)
        for name,row in counts(model,fast,groups).items():
            aggregate=total.setdefault(name,{'correct':0,'total':0})
            for field in aggregate:aggregate[field]+=row[field]
    model.train(was_training)
    return {name:{**row,'accuracy':row['correct']/row['total'] if row['total'] else None}
            for name,row in total.items()}
