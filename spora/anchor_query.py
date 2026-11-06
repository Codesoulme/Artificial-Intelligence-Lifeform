"""Learned causal question-target attention; labels are training-only."""
import torch
from torch import nn
from torch.nn import functional as F
from .query_bridge import ContextQueryBridge,extend_query
from .language_training import interface_state,load_interface

class AnchorQueryBridge(ContextQueryBridge):
    def __init__(self,organism):
        super().__init__(organism);d=organism.config.width
        self.anchor_norm=nn.LayerNorm(d)
        self.anchor_q=nn.Linear(d,d,bias=False);self.anchor_k=nn.Linear(d,d,bias=False)
        self.anchor_bytes=nn.Conv1d(d,d,3,padding=2)
        self.anchor_out=nn.Linear(d,d,bias=False)
        nn.init.zeros_(self.anchor_out.weight)
        self.anchor_logits=None

    def contextualize(self,x):
        n=x.shape[1];z=self.evidence_norm(x);a=self.anchor_norm(x)
        mask=torch.ones(n,n,device=x.device,dtype=torch.bool).tril()
        self.anchor_logits=(self.anchor_q(a)@self.anchor_k(a).transpose(-1,-2)/(x.shape[-1]**.5)).masked_fill(~mask,-torch.inf)
        raw=self._evidence_input.transpose(1,2)
        values=F.gelu(self.anchor_bytes(raw)[:,:,:n]).transpose(1,2)
        selected=self.anchor_logits.softmax(-1)@values
        local=F.gelu(self.local_query(raw)[:,:,:n]).transpose(1,2)
        q=F.normalize(local+self.context_query(x)+self.anchor_out(selected),dim=-1)
        k=F.normalize(F.gelu(self.local_key(raw)[:,:,:n]).transpose(1,2),dim=-1)
        self.content_logits=self.match_scale.exp().clamp(max=100.)*(q@k.transpose(-1,-2))
        positions=torch.arange(n,device=x.device,dtype=x.dtype)
        scores=self.content_logits+F.softplus(self.recency_logit)*(positions[None,:]-positions[:,None])
        self.evidence_logits=scores.masked_fill(~mask,-torch.inf)
        # Expose the selected target also to the learned copy/generate decision.
        self.copy_gate_logits=self.copy_gate(z+self.anchor_out(selected))
        return x+self.evidence_out(self.evidence_logits.softmax(-1)@self.evidence_v(z))

def extend_anchor(organism,payload,parent_sha256,device='cpu'):
    source=extend_query(organism,payload,parent_sha256,'cpu')
    model=AnchorQueryBridge(organism);merged=interface_state(model);merged.update(interface_state(source))
    load_interface(model,dict(format='spora-byte-interface-v1',organism_sha256=parent_sha256,state=merged),parent_sha256)
    return model.to(device).eval()

def restore_anchor(organism,payload,parent_sha256,device='cpu'):
    if payload.get('format')!='spora-anchor-query-v1' or payload.get('arm')!='context':raise ValueError('Not an anchored-query checkpoint')
    model=AnchorQueryBridge(organism)
    load_interface(model,dict(format='spora-byte-interface-v1',organism_sha256=payload['organism_sha256'],state=payload['state']),parent_sha256)
    return model.to(device).eval()
