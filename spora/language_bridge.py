"""Untrained UTF-8 byte interface sharing the organism's causal backbone.

Stage-one text updates touch only newly initialized interface parameters. The
organism and its persisted memory remain fixed; future alignment training must
teach semantic access. This file does not claim language ability.
"""
import torch
from torch import nn
from torch.nn import functional as F

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
    def __init__(self,organism):
        super().__init__();self.organism=organism
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
        x=self.byte_embedding(tokens)+self.byte_position(torch.arange(n,device=device))
        layer_site=0
        for i,block in enumerate(self.organism.blocks):
            x=block(x)
            if i in self.organism.config.memory_layers:
                q=F.normalize(self.query_projection(x),dim=-1)
                matrices=[state.current[layer_site]]+[page[layer_site][0] for page in state.pages]
                # Read-only access to value-bearing neural matrices; no raw observations.
                retrieved=torch.stack([torch.einsum('bmd,bnd->bnm',matrix.expand(b,-1,-1).detach(),q) for matrix in matrices],2)
                # Content-based mixture; page index is not used as an answer lookup.
                scores=(retrieved*q.unsqueeze(2)).sum(-1)/(q.shape[-1]**.5)
                page_keys=self.page_key(torch.arange(len(matrices),device=device))
                scores=scores+torch.einsum('bnm,pm->bnp',q,page_keys)/(q.shape[-1]**.5)
                mixed=(retrieved*scores.softmax(2).unsqueeze(-1)).sum(2)
                x=x+torch.sigmoid(self.read_strength)*self.read_projection(mixed)
                layer_site+=1
        x=x+self.adapter(x)
        return self.head(self.organism.final_norm(x))

    def trainable_parameters(self):return [p for p in self.parameters() if p.requires_grad]


def causal_loss(logits,tokens):
    if tokens.shape[1]<2:raise ValueError('At least two tokens required')
    return F.cross_entropy(logits[:,:-1].reshape(-1,VOCAB),tokens[:,1:].reshape(-1),ignore_index=PAD)
