"""Finite temporal associative memory. Queries never receive an event history."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import SPORA,ModelConfig,DeltaMemorySite


@dataclass
class TemporalConfig(ModelConfig):
    value_count:int=17
    memory_kind:str='delta'
    memory_width:int=90
    temporal_kind:str='split'
    time_count:int=64
    time_start:int=192

    @property
    def unknown(self):return self.value_count-1

    def validate(self):
        super().validate()
        if self.temporal_kind not in ('split','shared'):raise ValueError('Invalid temporal architecture')
        if self.memory_kind!='delta' or self.residual_write:raise ValueError('Temporal reference uses direct delta')
        if self.time_count<2 or self.time_start<self.value_start+self.value_count:
            raise ValueError('Time vocabulary overlaps values')
        if self.time_start+self.time_count>self.vocab_size:raise ValueError('Time vocabulary too large')


class TemporalSite(DeltaMemorySite):
    def __init__(self,config):
        super().__init__(config)
        self.split=config.temporal_kind=='split'
        if self.split:
            m=config.memory_width
            self.initial.append(nn.Parameter(torch.randn(m,m)*.02))

    def selected(self,params,mode):
        if not self.split:return params[0]
        return torch.where(mode[:,None,None].bool(),params[1],params[0])

    def read_temporal(self,x,query,params,mode):
        return super().read(x,query,[self.selected(params,mode)])

    def write_temporal(self,params,velocity,key,target,gate,mode,differentiable):
        matrix=self.selected(params,mode)
        update=gate[:,None,None]*torch.sigmoid(self.log_lr)*(target.detach()-self.retrieve([matrix],key)).unsqueeze(-1)*key.unsqueeze(1)
        updated=[]
        for i,p in enumerate(params):
            mask=(mode==i).to(p.dtype)[:,None,None] if self.split else 1.
            candidate=p+mask*update
            updated.append(candidate if differentiable else candidate.detach().requires_grad_(True))
        return updated,velocity


class TemporalSPORA(SPORA):
    def __init__(self,config=None):
        config=config or TemporalConfig()
        super().__init__(config)
        self.memories=nn.ModuleList([TemporalSite(config) for _ in config.memory_layers])
        self.history_binding=nn.Linear(config.width,config.width,bias=False)

    def write_event(self,*args,**kwargs):
        raise ValueError('Temporal models require an explicit event time; use write_temporal or spora.temporal_cli')

    def validate_query(self,keys,modes,times):
        if keys.ndim!=1 or modes.shape!=keys.shape or times.shape!=keys.shape:
            raise ValueError('Query vectors must have matching shape')
        if any(x.dtype!=torch.long for x in (keys,modes,times)):raise ValueError('Integer queries required')
        if bool(((keys<0)|(keys>=self.config.key_count)).any()):raise ValueError('Key out of range')
        if bool(((modes<0)|(modes>1)).any()):raise ValueError('Mode is current=0 or event=1')
        if bool(((times<0)|(times>=self.config.time_count)).any()):raise ValueError('Time out of range')

    def representation(self,keys,modes,times):
        self.validate_query(keys,modes,times)
        canonical=torch.where(modes.bool(),times,torch.zeros_like(times))
        tokens=torch.stack([torch.ones_like(keys),3+modes,
                            torch.where(modes.bool(),canonical+self.config.time_start,torch.full_like(keys,2)),
                            keys+self.config.key_start],dim=1)
        key=self.embedding(keys+self.config.key_start)
        time=self.embedding(canonical+self.config.time_start)
        history=self.history_binding(key*time*10.)
        bound=torch.where(modes[:,None].bool(),history,key)
        return tokens,bound

    def forward(self,keys,fast,modes=None,times=None):
        if modes is None:modes=torch.zeros_like(keys)
        if times is None:times=torch.zeros_like(keys)
        tokens,bound=self.representation(keys,modes,times)
        x=self.embedding(tokens)+self.position(torch.arange(tokens.shape[1],device=keys.device))
        site_index=0
        for index,block in enumerate(self.blocks):
            x=block(x)
            if index in self.config.memory_layers:
                site=self.memories[site_index]
                x=site.read_temporal(x,site.address(bound),fast[site_index],modes)
                site_index+=1
        c=self.config
        return F.linear(self.final_norm(x[:,-1]),self.embedding.weight[c.value_start:c.value_start+c.value_count])

    def predict_many(self,keys,fast,modes=None,times=None):
        b,n=keys.shape
        if modes is None:modes=torch.zeros_like(keys)
        if times is None:times=torch.zeros_like(keys)
        expanded=[[p.repeat_interleave(n,0) for p in site] for site in fast]
        return self(keys.flatten(),expanded,modes.flatten(),times.flatten()).reshape(b,n,-1)

    def write_temporal(self,fast,velocity,keys,values,times,trusted,differentiable=False,history=True):
        if values.shape!=keys.shape or trusted.shape!=keys.shape:raise ValueError('Bad event shape')
        if values.dtype!=torch.long or bool(((values<0)|(values>=self.config.unknown)).any()):
            raise ValueError('Cannot write unknown or invalid value')
        if not bool(torch.isfinite(trusted).all()) or bool(((trusted!=0)&(trusted!=1)).any()):
            raise ValueError('Trust must be a binary environment flag')
        self.validate_query(keys,torch.ones_like(keys),times)
        value=self.embedding(values+self.config.value_start)
        # Single observation, two explicit neural writes; counts are reported separately.
        for mode_index in (0,1) if history else (0,):
            modes=torch.full_like(keys,mode_index)
            _,bound=self.representation(keys,modes,times)
            gate=trusted if mode_index==0 else torch.ones_like(trusted)
            ff=[];vv=[]
            for site,params,moments in zip(self.memories,fast,velocity):
                p,u=site.write_temporal(params,moments,site.address(bound),site.target(value),gate,modes,differentiable)
                ff.append(p);vv.append(u)
            fast,velocity=ff,vv
        return fast,velocity


def reset_bank(model,fast,bank):
    if model.config.temporal_kind!='split':raise ValueError('Selective reset requires split architecture')
    if bank not in (0,1):raise ValueError('Bank must be current=0 or history=1')
    reset,_=model.initial_state(fast[0][0].shape[0])
    return [[reset[i][j] if j==bank else p.clone() for j,p in enumerate(site)] for i,site in enumerate(fast)]
