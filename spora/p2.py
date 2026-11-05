"""Correction curriculum, bounded writes, and streaming evaluation utilities."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .tasks import mappings


@dataclass
class Event:
    key: torch.Tensor
    value: torch.Tensor
    trusted: torch.Tensor
    kind: str


def stream(generator,batch_size,pairs,config,device,corrections=4):
    """Environment owns truth; inference never sees this table."""
    keys,original=mappings(generator,batch_size,pairs,config,device)
    current=original.clone()
    one=torch.ones(batch_size,device=device)
    zero=torch.zeros(batch_size,device=device)
    events=[Event(keys[:,i],original[:,i],one,'new') for i in range(pairs)]
    for i in range(min(corrections,pairs)):
        # Balanced nonzero shifts, not a learned fixed 'plus one' label shortcut.
        shift=torch.randint(1,config.value_count,(batch_size,),generator=generator).to(device)
        current[:,i]=(current[:,i]+shift)%config.value_count
        events.append(Event(keys[:,i],current[:,i],one,'correction'))
        events.append(Event(keys[:,i],(current[:,i]+1)%config.value_count,zero,'untrusted'))
        events.append(Event(keys[:,i],current[:,i],one,'repeat'))
    return events,keys,original,current


def detach_state(fast,velocity):
    return ([[p.detach().requires_grad_(True) for p in site] for site in fast],
            [[u.detach() for u in site] for site in velocity])


def write(model,fast,velocity,event,policy='one',threshold=.8,differentiable=False):
    """One externally observed event, up to four bounded internal updates."""
    if policy not in ('one','two','adaptive-max4'):
        raise ValueError('Unknown write policy')
    maximum={'one':1,'two':2,'adaptive-max4':4}[policy]
    if not 0<threshold<1: raise ValueError('Invalid adaptive threshold')
    active=event.trusted.to(dtype=model.embedding.weight.dtype).clone()
    steps=torch.zeros_like(active)
    for _ in range(maximum):
        if not bool(active.any()): break
        fast,velocity=model.write_event(fast,velocity,event.key,event.value,
                                       differentiable=differentiable,trusted=active)
        steps=steps+active
        if policy=='adaptive-max4':
            with torch.no_grad():
                probability=model(event.key,fast).softmax(-1).gather(1,event.value[:,None]).squeeze(1)
                active=active*(probability<threshold).to(active.dtype)
    return fast,velocity,steps


def geometry_loss(model):
    """A soft address-decorrelation candidate, not a protection guarantee."""
    keys=torch.arange(model.config.key_count,device=model.embedding.weight.device)
    embedding=model.embedding(keys+model.config.key_start)
    loss=0.
    for site in model.memories:
        q=F.normalize(site.address(embedding),dim=-1)
        gram=q@q.T
        mask=~torch.eye(len(keys),device=keys.device,dtype=torch.bool)
        loss=loss+gram[mask].square().mean()
    return loss/len(model.memories)


def world_loss(model,events,all_keys,fast,velocity,policy,threshold,segment_length=8):
    """Keep theta fixed throughout a world; truncate only fast-state gradients."""
    truth={}
    losses=[]
    segments=(len(events)+segment_length-1)//segment_length
    updates=0.
    for start in range(0,len(events),segment_length):
        end=min(start+segment_length,len(events))
        correction_losses=[]
        for event in events[start:end]:
            fast,velocity,steps=write(model,fast,velocity,event,policy,threshold,True)
            updates+=float(steps.sum().detach())
            if event.kind=='correction':
                correction_losses.append(F.cross_entropy(model(event.key,fast),event.value))
            if event.kind!='untrusted':
                # Rows in a batch have different key order, but share logical column.
                positions=(all_keys==event.key[:,None]).to(torch.int64).argmax(1)
                for row,pos in enumerate(positions.tolist()):
                    truth[row,pos]=event.value[row]
        positions=sorted({pos for row,pos in truth if row==0})
        query=all_keys[:,positions]
        targets=torch.stack([torch.stack([truth[row,pos] for pos in positions])
                             for row in range(all_keys.shape[0])])
        task=F.cross_entropy(model.predict_many(query,fast).flatten(0,1),targets.flatten())
        if correction_losses:
            task=task+torch.stack(correction_losses).mean()
        (task/segments).backward()
        losses.append(float(task.detach()))
        fast,velocity=detach_state(fast,velocity)
    return sum(losses)/len(losses),updates


def evaluate(model,worlds=32,pairs=8,seed=510000,policy='one',threshold=.8,protection='none'):
    model.eval()
    generator=torch.Generator().manual_seed(seed)
    device=model.embedding.weight.device
    totals={k:0. for k in ['initial','corrected','unrelated','final','oldest8','latest8',
                             'untrusted_unchanged','trusted_writes','inner_updates']}
    denominators={k:0 for k in totals}
    for offset in range(0,worlds,8):
        b=min(8,worlds-offset)
        events,keys,original,current=stream(generator,b,pairs,model.config,device)
        fast,velocity=model.initial_state(b)
        written={}
        for index,event in enumerate(events):
            if event.kind=='untrusted':
                before=[[p.clone() for p in site] for site in fast]
            fast,velocity,steps=write(model,fast,velocity,event,policy,threshold)
            totals['inner_updates']+=float(steps.sum())
            if event.kind!='untrusted':
                totals['trusted_writes']+=b
                for row in range(b):written[row,int(event.key[row])]=int(event.value[row])
            if protection=='replay' and event.kind=='correction':
                # Training-only rehearsal; choose an old unrelated key, excluding stale target.
                replay_pos=min(pairs-1,4)
                replay_key=keys[:,replay_pos]
                replay_value=torch.tensor([written[row,int(replay_key[row])] for row in range(b)],device=device)
                replay=Event(replay_key,replay_value,torch.ones(b,device=device),'replay')
                fast,velocity,extra=write(model,fast,velocity,replay,policy,threshold)
                totals['inner_updates']+=float(extra.sum())
            if event.kind=='untrusted':
                same=all(torch.equal(a,c) for left,right in zip(before,fast) for a,c in zip(left,right))
                totals['untrusted_unchanged']+=b*same;denominators['untrusted_unchanged']+=b
            if index==pairs-1:
                with torch.no_grad():pred=model.predict_many(keys,fast).argmax(-1)
                totals['initial']+=float((pred==original).sum());denominators['initial']+=b*pairs
            if event.kind=='correction':
                with torch.no_grad():pred=model(event.key,fast).argmax(-1)
                totals['corrected']+=float((pred==event.value).sum());denominators['corrected']+=b
        with torch.no_grad():pred=model.predict_many(keys,fast).argmax(-1)
        matches=(pred==current)
        for name,slice_ in [('final',slice(None)),('oldest8',slice(0,8)),('latest8',slice(-8,None)),('unrelated',slice(4,None))]:
            selected=matches[:,slice_]
            totals[name]+=float(selected.sum());denominators[name]+=selected.numel()
    result={k:totals[k]/denominators[k] for k in totals if denominators[k]}
    result['inner_updates_per_trusted_event']=totals['inner_updates']/totals['trusted_writes']
    result['worlds']=worlds;result['pairs']=pairs
    model.train()
    return result
