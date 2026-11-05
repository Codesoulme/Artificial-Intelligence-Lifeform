"""Trainable language cortex with read-only access to the same neural memories."""
import torch
from torch import nn
from .model import CausalBlock
from .language_bridge import LanguageBridge,VOCAB
from torch.nn import functional as F
from .language_training import load_interface


class CortexLanguageBridge(LanguageBridge):
    def __init__(self, organism, source=None, parent_sha256=None, depth=2, local_bytes=False, evidence_reader=False, copy_reader=False, local_evidence=False,query_window=32):
        super().__init__(organism)
        if depth not in (2,4):raise ValueError('Supported language depth is 2 or 4')
        self.depth=depth
        if len(organism.config.memory_layers)!=2:
            raise ValueError('Two memory sites required by this research architecture')
        if source is not None:load_interface(self,source,parent_sha256)
        self.cortex=nn.ModuleList([CausalBlock(organism.config) for _ in range(depth)])
        self.local_bytes=local_bytes
        if local_bytes:
            self.byte_composer=nn.Conv1d(organism.config.width,organism.config.width,3,padding=2)
        self.evidence_reader=evidence_reader
        self.copy_reader=copy_reader
        self.local_evidence=local_evidence
        if query_window not in (24,32):raise ValueError("Supported query windows:24/32")
        self.query_window=query_window
        if local_evidence and not copy_reader:raise ValueError('Local evidence requires copying')
        if copy_reader and not evidence_reader:raise ValueError('Copying needs learned evidence attention')
        if evidence_reader:
            d=organism.config.width
            self.evidence_norm=nn.LayerNorm(d)
            self.evidence_q=nn.Linear(d,d,bias=False)
            self.evidence_k=nn.Linear(d,d,bias=False)
            self.evidence_v=nn.Linear(d,d,bias=False)
            self.evidence_out=nn.Linear(d,d,bias=False)
            nn.init.zeros_(self.evidence_out.weight)
            if copy_reader:
                self.copy_gate=nn.Linear(d,1)
                nn.init.zeros_(self.copy_gate.weight);nn.init.constant_(self.copy_gate.bias,-2.)
            if local_evidence:
                self.local_query=nn.Conv1d(d,d,query_window,padding=query_window-1)
                self.local_key=nn.Conv1d(d,d,9,padding=8)
                self.match_scale=nn.Parameter(torch.tensor(3.))
                self.recency_logit=nn.Parameter(torch.tensor(-5.3))
        self.evidence_logits=None
        self.content_logits=None
        self.copy_gate_logits=None
        # Disclosed new-language positional initialization; all other interface
        # weights inherit P18. The original positional parameters remain intact.
        nn.init.normal_(self.byte_position.weight,std=.02)

    def language_blocks(self):return self.cortex

    def language_memory_layers(self):return (self.depth//2-1,self.depth-1)

    def contextualize(self,x):
        if not self.evidence_reader:return x
        z=self.evidence_norm(x);n=x.shape[1]
        scores=self.evidence_q(z)@self.evidence_k(z).transpose(-1,-2)/(x.shape[-1]**.5)
        if self.local_evidence:
            # Shared learned windows prevent evidence keys from encoding an
            # entire object pair. No parsing or answer labels enter inference.
            raw=self._evidence_input.transpose(1,2)
            q=F.normalize(F.gelu(self.local_query(raw)[:,:,:n]).transpose(1,2),dim=-1)
            k=F.normalize(F.gelu(self.local_key(raw)[:,:,:n]).transpose(1,2),dim=-1)
            positions=torch.arange(n,device=x.device,dtype=x.dtype)
            scores=self.match_scale.exp().clamp(max=100.)*(q@k.transpose(-1,-2))
            self.content_logits=scores
            # Explicit disclosed monotone recency prior; its strength is learned.
            scores=scores+F.softplus(self.recency_logit)*(positions[None,:]-positions[:,None])
        mask=torch.ones(n,n,device=x.device,dtype=torch.bool).tril()
        self.evidence_logits=scores.masked_fill(~mask,-torch.inf)
        if self.copy_reader:self.copy_gate_logits=self.copy_gate(z)
        return x+self.evidence_out(self.evidence_logits.softmax(-1)@self.evidence_v(z))

    def forward(self,tokens,state):
        if self.local_evidence:self._evidence_input=self.byte_embedding(tokens)
        logits=super().forward(tokens,state)
        if not self.copy_reader:return logits
        attention=self.evidence_logits.softmax(-1)
        copied=torch.zeros_like(logits).scatter_add(2,tokens[:,None,:].expand(-1,tokens.shape[1],-1),attention)
        # Both branches are learned. No task parser or answer set is consulted.
        return torch.logaddexp(F.log_softmax(logits,-1)+F.logsigmoid(-self.copy_gate_logits),
            copied.clamp_min(1e-30).log()+F.logsigmoid(self.copy_gate_logits))

    def embed_tokens(self,tokens):
        if not self.local_bytes:return super().embed_tokens(tokens)
        x=self.byte_embedding(tokens)
        local=self.byte_composer(x.transpose(1,2))[:,:,:tokens.shape[1]].transpose(1,2)
        return x+torch.nn.functional.gelu(local)+self.byte_position(torch.arange(tokens.shape[1],device=tokens.device))


def restore_ladder(organism,payload,parent_sha256,device='cpu'):
    if payload.get('format')!='spora-language-ladder-v1':raise ValueError('Not a ladder checkpoint')
    arm=payload.get('arm')
    if arm not in ('original','cortex'):raise ValueError('Unknown language architecture')
    bridge=CortexLanguageBridge(organism,depth=payload.get('depth',2),local_bytes=payload.get('local_bytes',False),evidence_reader=payload.get('evidence_reader',False),copy_reader=payload.get('copy_reader',False),local_evidence=payload.get('local_evidence',False),query_window=payload.get('query_window',32)) if arm=='cortex' else LanguageBridge(organism)
    load_interface(bridge,{'format':'spora-byte-interface-v1','organism_sha256':payload['organism_sha256'],
        'state':payload['state']},parent_sha256)
    return bridge.to(device).eval()
