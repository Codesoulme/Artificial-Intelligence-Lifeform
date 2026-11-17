"""Neural record selection prototype with an explicit null candidate."""
import torch
from torch import nn
from torch.nn import functional as F


class IdentitySelector(nn.Module):
    def __init__(self, width=1536, rank=64):
        super().__init__()
        self.query=nn.Linear(width,rank,bias=False)
        self.record=nn.Linear(width,rank,bias=False)
        self.null_score=nn.Parameter(torch.tensor(0.0))

    def forward(self, query, records):
        if query.ndim!=2 or records.ndim!=3 or query.shape[0]!=records.shape[0] or query.shape[-1]!=records.shape[-1]:
            raise ValueError('Invalid query or record representations')
        q=F.normalize(self.query(F.layer_norm(query.float(),(query.shape[-1],))),dim=-1)
        r=F.normalize(self.record(F.layer_norm(records.float(),(records.shape[-1],))),dim=-1)
        logits=12.0*(q.unsqueeze(1)*r).sum(-1)
        return torch.cat((logits,self.null_score.expand(query.shape[0],1)),dim=-1)
