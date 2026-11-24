"""Match a scoped turn to draft representations without rendering draft text into the turn."""

import torch
from torch import nn
from torch.nn import functional as F


class DraftReferenceMatcher(nn.Module):
    """Score each pending draft and an explicit unresolved outcome."""

    def __init__(self, hidden_width, width=64):
        super().__init__()
        self.query = nn.Linear(hidden_width, width)
        self.draft = nn.Linear(hidden_width, width)
        self.unresolved = nn.Linear(width, 1)

    def forward(self, turn, drafts):
        query = self.query(F.normalize(turn.float(), dim=-1))
        keys = self.draft(F.normalize(drafts.float(), dim=-1))
        similarity = (F.normalize(keys, dim=-1) * F.normalize(query, dim=-1).unsqueeze(1)).sum(dim=-1)
        return torch.cat((12. * similarity, self.unresolved(query)), dim=-1)
