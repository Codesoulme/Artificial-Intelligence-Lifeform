"""Untrained UTF-8 byte interface sharing the organism's causal backbone.

Stage-one text updates touch only newly initialized interface parameters. The
organism and its persisted memory remain fixed; future alignment training must
teach semantic access. This file does not claim language ability.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .memory_read import MemoryReadPlan
from .sleep_memory import history_site

BOS=256
EOS=257
PAD=258
VOCAB=259


def encode(text):
    if not isinstance(text,str):raise TypeError('Text required')
    return [BOS]+list(text.encode('utf-8'))+[EOS]


def decode(tokens):
    return bytes([x for x in tokens if 0<=x<256]).decode('utf-8',errors='strict')


class LanguageBridge(nn.Module):
    def __init__(self,organism,read_mode='exact',read_error_l2=1e-4):
        super().__init__();self.organism=organism
        if read_mode not in ('legacy','exact','bounded','learned'):raise ValueError('Invalid read mode')
        self.read_mode=read_mode;self.read_error_l2=read_error_l2
        self._read_plans={};self.last_read_stats=[]
        for p in organism.parameters():p.requires_grad_(False)
        d=organism.config.width;m=organism.config.memory_width
        self.byte_embedding=nn.Embedding(VOCAB,d,padding_idx=PAD)
        self.byte_position=nn.Embedding(organism.config.max_length,d)
        self.query_projection=nn.Linear(d,m)
        self.read_projection=nn.Linear(m,d)
        self.page_key=nn.Embedding(17,m)
        self.read_strength=nn.Parameter(torch.tensor(-2.))
        self.adapter=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
        self.head=nn.Linear(d,VOCAB,bias=False)
        nn.init.normal_(self.byte_embedding.weight,std=.02)

    def forward(self,tokens,state):
        if tokens.ndim!=2 or tokens.dtype!=torch.long:raise ValueError('Batch of integer byte tokens required')
        if tokens.shape[1]<1 or tokens.shape[1]>self.organism.config.max_length:raise ValueError('Context extent invalid')
        if bool(((tokens<0)|(tokens>=VOCAB)).any()):raise ValueError('Byte token out of range')
        b,n=tokens.shape;device=tokens.device
        self.last_read_stats=[]
        x=self.embed_tokens(tokens)
        layer_site=0
        for i,block in enumerate(self.language_blocks()):
            x=block(x)
            if i in self.language_memory_layers():
                q=F.normalize(self.query_projection(x),dim=-1)
                matrices=[state.current[layer_site]]+[history_site(page,layer_site)[0] for page in state.pages]
                # Read-only access to value-bearing neural matrices; no raw observations.
                page_keys=self.page_key(torch.arange(len(matrices),device=device))
                # Preserve the existing gradient path for all training. Prepared
                # read plans are derived, disposable inference state only.
                unversioned=any(torch.is_inference(matrix) for matrix in matrices)
                if torch.is_grad_enabled() or self.read_mode=='legacy' or unversioned:
                    retrieved=torch.stack([torch.einsum('bmd,bnd->bnm',matrix.expand(b,-1,-1).detach(),q) for matrix in matrices],2)
                    scores=(retrieved*q.unsqueeze(2)).sum(-1)/(q.shape[-1]**.5)
                    scores=scores+torch.einsum('bnm,pm->bnp',q,page_keys)/(q.shape[-1]**.5)
                    mixed=(retrieved*scores.softmax(2).unsqueeze(-1)).sum(2)
                    stats={'mode':'training' if torch.is_grad_enabled() else ('legacy-unversioned' if unversioned else 'legacy'),
                           'matrix_vectors':b*n*len(matrices),'baseline_matrix_vectors':b*n*len(matrices)}
                else:
                    factors=None
                    mode=self.read_mode
                    if mode=='learned':
                        if state.internal is not None and state.internal.router_factors is not None and state.internal.router_cursor==state.cursor:
                            factors=state.internal.router_factors[layer_site]
                        else:mode='exact'
                    plan=self._read_plans.get(layer_site)
                    rebuild=plan is None or not plan.matches(matrices)
                    if factors is not None:
                        from .learned_read import LearnedReadPlan
                        rebuild=rebuild or not isinstance(plan,LearnedReadPlan) or not plan.matches_factors(factors)
                        if rebuild:plan=LearnedReadPlan(matrices,factors)
                    elif rebuild:plan=MemoryReadPlan(matrices)
                    self._read_plans[layer_site]=plan
                    mixed,stats=plan.read(q,page_keys,mode,self.read_error_l2)
                    if self.read_mode=='learned' and factors is None:stats['router_unavailable']=True
                    stats.update(plan_rebuilt=rebuild,plan_tensor_bytes=plan.tensor_bytes)
                self.last_read_stats.append(stats)
                x=x+torch.sigmoid(self.read_strength)*self.read_projection(mixed)
                layer_site+=1
        x=self.contextualize(x)
        x=x+self.adapter(x)
        return self.head(self.organism.final_norm(x))

    def contextualize(self,x):
        """Identity in the original interface; optional research extension point."""
        return x

    def language_blocks(self):
        return self.organism.blocks

    def embed_tokens(self,tokens):
        return self.byte_embedding(tokens)+self.byte_position(torch.arange(tokens.shape[1],device=tokens.device))

    def language_memory_layers(self):
        return self.organism.config.memory_layers

    def trainable_parameters(self):return [p for p in self.parameters() if p.requires_grad]


def causal_loss(logits,tokens):
    if tokens.shape[1]<2:raise ValueError('At least two tokens required')
    return F.cross_entropy(logits[:,:-1].reshape(-1,VOCAB),tokens[:,1:].reshape(-1),ignore_index=PAD)
