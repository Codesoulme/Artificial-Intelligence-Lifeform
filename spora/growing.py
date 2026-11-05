"""Explicit append-only neural history pages, with one global current memory.

Page routing is a disclosed time-range prior. No raw key/value observations are
stored. Capacity and persisted inverse covariance are fully charged.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib,os
import torch
from .temporal import TemporalSPORA,TemporalConfig
from .sleep_memory import history_site,SealedPage


@dataclass
class GrowingState:
    current:list
    pages:list
    cursor:int=0
    runtime:object=None


def new_state(model):
    if model.config.temporal_kind!='split':raise ValueError('Growing history requires split architecture')
    f,_=model.initial_state(1)
    return GrowingState([s[0] for s in f],[[s[1:] for s in f]])


def capacity(model,state):return len(state.pages)*model.config.time_count


def grow(model,state,pages=1,max_pages=16):
    if type(pages) is not int or pages<1 or type(max_pages) is not int or not 1<=max_pages<=16 or len(state.pages)+pages>max_pages:
        raise ValueError('Invalid or excessive growth request')
    added=[]
    for _ in range(pages):
        f,_=model.initial_state(1);added.append([s[1:] for s in f])
    return GrowingState(state.current,state.pages+added,state.cursor,state.runtime)


def observe(model,state,key,value,trusted,time=None):
    if time is not None and (type(time) is not int or time!=state.cursor):
        raise ValueError('Observation time must be the next cursor; no rewrites or gaps')
    if state.cursor>=capacity(model,state):raise ValueError('Explicit growth required; original state unchanged')
    device=next(model.parameters()).device
    page=state.cursor//model.config.time_count;local=state.cursor%model.config.time_count
    if not isinstance(state.pages[page],list):raise ValueError('Sealed history cannot accept new writes')
    fast=[[current]+history for current,history in zip(state.current,state.pages[page])]
    velocity=[[torch.zeros_like(p) for p in site] for site in fast]
    fast,_=model.write_temporal(fast,velocity,torch.tensor([key],device=device),torch.tensor([value],device=device),
                              torch.tensor([local],device=device),torch.tensor([trusted],dtype=torch.float32,device=device))
    pages=list(state.pages);pages[page]=[s[1:] for s in fast]
    return GrowingState([s[0] for s in fast],pages,state.cursor+1,state.runtime)


def query(model,state,keys,modes,times):
    if keys.ndim!=1 or keys.shape!=modes.shape or keys.shape!=times.shape:raise ValueError('Matching vectors required')
    if keys.numel()==0:raise ValueError('At least one query required')
    if times.dtype!=torch.long or bool(((times<0)|(times>=capacity(model,state))).any()):raise ValueError('Time outside allocated range')
    indices=(times//model.config.time_count).tolist();batch=keys.numel()
    fast=[]
    for site,current in enumerate(state.current):
        selected={p:history_site(state.pages[p],site) for p in set(indices)}
        history=[torch.cat([selected[p][j] for p in indices],0) for j in range(len(model.memories[site].initial)-1)]
        fast.append([current.expand(batch,*current.shape[1:])]+history)
    return model(keys,fast,modes,times%model.config.time_count)


def scalars(state):
    current=sum(p.numel() for p in state.current)
    history=sum(page.scalar_count() if isinstance(page,SealedPage) else sum(p.numel() for site in page for p in site) for page in state.pages)
    return {'current':current,'history_including_optimizer_evidence':history,'total':current+history}


def save(path,model,state,metadata=None):
    if state.runtime is not None or hasattr(model,'time_world') or any(isinstance(page,SealedPage) for page in state.pages):
        from .state_codec import flatten
        schema,tensors=flatten(model,state)
        if not all(bool(torch.isfinite(x).all()) for x in tensors.values()):raise ValueError('Nonfinite state; preserve old file')
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
        data={'format':'spora-growing-v2','schema':schema,'tensors':{k:v.detach().cpu().clone() for k,v in tensors.items()},'metadata':metadata or {}}
        try:
            with temporary.open('wb') as stream:torch.save(data,stream);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,path)
        finally:
            if temporary.exists():temporary.unlink()
        return
    if model.config.temporal_kind!='split' or not 1<=len(state.pages)<=16:
        raise ValueError('Growing state requires one to sixteen split pages')
    if type(state.cursor) is not int:raise ValueError('Integer cursor required')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data={'format':'spora-growing-v1','config':model.config_dict(),
          'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
          'current':[x.detach().cpu().clone() for x in state.current],
          'pages':[[[x.detach().cpu().clone() for x in site] for site in page] for page in state.pages],
          'cursor':state.cursor,'metadata':metadata or {}}
    tensors=list(data['model'].values())+data['current']+[x for p in data['pages'] for s in p for x in s]
    if not all(bool(torch.isfinite(x).all()) for x in tensors):raise ValueError('Nonfinite state; preserve old file')
    if not 0<=state.cursor<=capacity(model,state):raise ValueError('Invalid cursor')
    temporary=path.with_suffix(path.suffix+'.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(data,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def load(path,device='cpu'):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if data.get('format')=='spora-growing-v2':
        from .state_codec import restore
        model,state=restore(data['schema'],data['tensors'],device)
        return model,state,data['metadata']
    if data.get('format')!='spora-growing-v1':raise ValueError('Unknown growing format')
    model=TemporalSPORA(TemporalConfig(**data['config'])).to(device);model.load_state_dict(data['model']);model.eval()
    if model.config.temporal_kind!='split':raise ValueError('Growing state must use split architecture')
    current=[x.to(device).detach() for x in data['current']]
    pages=[[[x.to(device).detach() for x in s] for s in p] for p in data['pages']]
    state=GrowingState(current,pages,data['cursor'])
    if type(state.cursor) is not int or not 1<=len(pages)<=16 or not 0<=state.cursor<=capacity(model,state):raise ValueError('Invalid growing extent')
    if len(current)!=len(model.memories):raise ValueError('Current layer mismatch')
    for page in pages:
        if len(page)!=len(model.memories):raise ValueError('Page layer mismatch')
        for i,(site,params) in enumerate(zip(model.memories,page)):
            if len(params)!=len(site.initial)-1:raise ValueError('History state missing')
            for tensor,initial in zip([current[i]]+params,site.initial):
                if tensor.shape!=(1,*initial.shape) or not bool(torch.isfinite(tensor).all()):raise ValueError('Bad tensor')
    if not all(bool(torch.isfinite(x).all()) for x in model.state_dict().values()):raise ValueError('Nonfinite model')
    return model,state,data['metadata']


def migrate(source,destination):
    """Adopt a P5 temporal individual's actual matrices, without replaying events."""
    from .checkpoint import load_checkpoint
    if Path(destination).exists():raise FileExistsError(destination)
    model,fast,_,data=load_checkpoint(source)
    if model.config.temporal_kind!='split' or fast[0][0].shape[0]!=1:raise ValueError('Split singleton required')
    state=GrowingState([s[0] for s in fast],[[s[1:] for s in fast]],data['cursor'])
    save(destination,model,state,{'parent_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                                'stage':'P6 exact state adoption','parent_stage':data['metadata'].get('stage')})
    return model,state
