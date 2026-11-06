"""P22 contextual residual query; no task parsing or labels in inference."""
import torch
from torch import nn
from torch.nn import functional as F
from .language_ladder import CortexLanguageBridge
from .language_training import load_interface


class ContextQueryBridge(CortexLanguageBridge):
    def __init__(self, organism):
        super().__init__(organism, depth=4, local_bytes=True, evidence_reader=True,
                         copy_reader=True, local_evidence=True, query_window=24)
        d=organism.config.width
        self.context_query=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d))
        # Exact function-preserving extension of a loaded P21 interface.
        nn.init.zeros_(self.context_query[-1].weight)
        nn.init.zeros_(self.context_query[-1].bias)

    def contextualize(self,x):
        z=self.evidence_norm(x);n=x.shape[1]
        raw=self._evidence_input.transpose(1,2)
        local=F.gelu(self.local_query(raw)[:,:,:n]).transpose(1,2)
        q=F.normalize(local+self.context_query(x),dim=-1)
        k=F.normalize(F.gelu(self.local_key(raw)[:,:,:n]).transpose(1,2),dim=-1)
        self.content_logits=self.match_scale.exp().clamp(max=100.)*(q@k.transpose(-1,-2))
        positions=torch.arange(n,device=x.device,dtype=x.dtype)
        scores=self.content_logits+F.softplus(self.recency_logit)*(positions[None,:]-positions[:,None])
        mask=torch.ones(n,n,device=x.device,dtype=torch.bool).tril()
        self.evidence_logits=scores.masked_fill(~mask,-torch.inf)
        self.copy_gate_logits=self.copy_gate(z)
        return x+self.evidence_out(self.evidence_logits.softmax(-1)@self.evidence_v(z))


def extend_query(organism,payload,parent_sha256,device='cpu'):
    from .language_ladder import restore_ladder
    # Validate the source with its own strict loader before merging new tensors.
    source=restore_ladder(organism,payload,parent_sha256,'cpu')
    from .language_training import interface_state
    model=ContextQueryBridge(organism)
    merged=interface_state(model);merged.update(interface_state(source))
    load_interface(model,dict(format='spora-byte-interface-v1',
                   organism_sha256=parent_sha256,state=merged),parent_sha256)
    return model.to(device).eval()


def restore_query(organism,payload,parent_sha256,device='cpu'):
    if payload.get('format')!='spora-context-query-v1':raise ValueError('Not a contextual query checkpoint')
    if payload.get('arm') not in ('local','context'):raise ValueError('Unknown query arm')
    model=ContextQueryBridge(organism) if payload['arm']=='context' else CortexLanguageBridge(
        organism,depth=4,local_bytes=True,evidence_reader=True,copy_reader=True,local_evidence=True,query_window=24)
    load_interface(model,dict(format='spora-byte-interface-v1',organism_sha256=payload['organism_sha256'],
        state=payload['state']),parent_sha256)
    return model.to(device).eval()
