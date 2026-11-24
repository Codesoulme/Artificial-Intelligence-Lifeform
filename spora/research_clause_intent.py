"""Read a sequence of clause-end representations for memory write decisions."""

import re

import torch
from torch import nn
from torch.nn import functional as F


_CLAUSE_BREAK = re.compile(r'(?<=[。！？；，,.!?;\n])(?=[^。！？；，,.!?;\n])')


def clause_ends(text):
    if not text:
        raise ValueError('A request must contain text')
    return [match.start() - 1 for match in _CLAUSE_BREAK.finditer(text)] + [len(text) - 1]


class ClauseIntent(nn.Module):
    def __init__(self, width, *, recurrent, classes=2):
        super().__init__()
        self.project = nn.Linear(width, 64, bias=False)
        self.recurrent = nn.GRU(64, 64, batch_first=True) if recurrent else None
        self.decision = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, classes))

    def forward(self, hidden, mask):
        hidden = F.gelu(self.project(F.normalize(hidden.float(), dim=-1)))
        lengths = mask.sum(-1).clamp_min(1)
        pooled = hidden.masked_fill(~mask[..., None], 0.).sum(1) / lengths[:, None]
        steps = hidden if self.recurrent is None else self.recurrent(hidden)[0]
        last = steps[torch.arange(len(hidden), device=hidden.device), lengths - 1]
        return self.decision(torch.cat((pooled, last), dim=-1))
