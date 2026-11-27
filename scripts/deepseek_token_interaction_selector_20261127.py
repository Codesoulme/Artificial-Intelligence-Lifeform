"""Experimental selector with per-token question interaction."""
import torch
import torch.nn.functional as F
from scripts.deepseek_query_evidence_probe_20261127 import Selector, DIM

class TokenInteractionSelector(Selector):
    def forward(self,hidden,query):
        if query.ndim != 2: raise ValueError('Expected all question token states')
        x=F.layer_norm(hidden.float(),(hidden.shape[-1],))
        q=F.layer_norm(query.float(),(query.shape[-1],))
        scores=(self.key(x)@self.query(q).T).max(dim=1).values/DIM**0.5
        return torch.cat((scores,self.null(q[-1]).reshape(1)))
