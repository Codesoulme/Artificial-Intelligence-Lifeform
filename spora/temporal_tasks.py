"""Environment-only records and independent query labels for the finite experiment."""
import torch


def events(generator,batch,length,config,device,offset=None):
    if not 4<=length<=config.time_count:raise ValueError('Invalid event count')
    # Four entities are revisited throughout; order varies by world.
    pool=torch.stack([torch.randperm(config.key_count,generator=generator)[:4] for _ in range(batch)])
    choices=torch.randint(4,(batch,length),generator=generator)
    choices[:,:4]=torch.arange(4)
    keys=pool.gather(1,choices)
    values=torch.randint(config.unknown,(batch,length),generator=generator)
    trusted=(torch.rand(batch,length,generator=generator)>.25).float();trusted[:,:4]=1
    # Each stream contains a trusted correction, an untrusted conflict, and a repeat.
    if length>=8:
        keys[:,4:8]=keys[:,0,None]
        values[:,4]=(values[:,0]+torch.randint(1,config.unknown,(batch,),generator=generator))%config.unknown
        trusted[:,4]=1;values[:,5]=(values[:,4]+3)%config.unknown;trusted[:,5]=0
        values[:,6]=values[:,4];trusted[:,6]=1;trusted[:,7]=1
    max_offset=config.time_count-length
    offsets=torch.randint(max_offset+1,(batch,),generator=generator) if offset is None else torch.full((batch,),offset)
    times=offsets[:,None]+torch.arange(length)[None,:]
    return {k:v.to(device) for k,v in {'keys':keys,'values':values,'trusted':trusted,'times':times}.items()}


def queries(record,end,config):
    keys=record['keys'][:,:end];values=record['values'][:,:end];trust=record['trusted'][:,:end];times=record['times'][:,:end]
    b=keys.shape[0];device=keys.device
    allkeys=torch.arange(config.key_count,device=device)[None,:].expand(b,-1)
    current=torch.full_like(allkeys,config.unknown)
    for i in range(end):
        old=current.gather(1,keys[:,i,None]).squeeze(1)
        current.scatter_(1,keys[:,i,None],torch.where(trust[:,i].bool(),values[:,i],old)[:,None])
    # Current queries include all four seen entities plus four guaranteed unobserved keys.
    current_keys=torch.cat([keys[:,:4],torch.stack([allkeys[r][current[r]==config.unknown][:4] for r in range(b)])],1)
    current_targets=current.gather(1,current_keys)
    invalid=(keys+1)%config.key_count
    # A time is globally unique; another key at that time must be unknown.
    next_time=(times[:,-1:]+1).clamp(max=config.time_count-1)
    future_keys=keys[:,:4]
    future_times=next_time.expand_as(future_keys)
    has_future=times[:,-1]<config.time_count-1
    return {
        'current':(current_keys,torch.zeros_like(current_keys),torch.zeros_like(current_keys),current_targets),
        'history':(keys,torch.ones_like(keys),times,values),
        'invalid':(invalid,torch.ones_like(keys),times,torch.full_like(values,config.unknown)),
        'future':(future_keys,torch.ones_like(future_keys),future_times,torch.full_like(future_keys,config.unknown)),
        'future_mask':has_future,
        'history_trust':trust,
    }
