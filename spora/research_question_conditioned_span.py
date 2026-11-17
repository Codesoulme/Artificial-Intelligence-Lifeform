"""Research-only candidate-specific interaction with an explicit question mask."""
import math
import torch
from torch import nn


class QuestionConditionedSpan(nn.Module):
    """Add a zero-initialized question interaction to a supplied span head."""

    def __init__(self, base, width, rank=32):
        super().__init__()
        if not isinstance(base, nn.Module) or width <= 0 or rank <= 0:
            raise ValueError('Invalid head dimensions')
        self.base = base
        self.question = nn.Linear(width, rank, bias=False)
        self.start = nn.Linear(width, rank, bias=False)
        self.end = nn.Linear(width, rank, bias=False)
        self.rank = rank
        nn.init.zeros_(self.start.weight)
        nn.init.zeros_(self.end.weight)

    def forward(self, features, mask, question_mask):
        if (features.ndim != 3 or mask.shape != features.shape[:2]
                or question_mask.shape != mask.shape or mask.dtype != torch.bool
                or question_mask.dtype != torch.bool or not bool(mask.any(1).all())
                or not bool(question_mask.any(1).all()) or bool((question_mask & ~mask).any())):
            raise ValueError('Question mask must select valid nonempty tokens')
        baseline = self.base(features, mask)
        pooled = (features * question_mask.unsqueeze(-1)).sum(1) / question_mask.sum(1, keepdim=True)
        question = self.question(pooled).unsqueeze(1)
        start = (self.start(features) * question).sum(-1) / math.sqrt(self.rank)
        end = (self.end(features) * question).sum(-1) / math.sqrt(self.rank)
        delta = (start[:, :, None] + end[:, None, :]).flatten(1)
        return baseline + torch.cat([torch.zeros_like(baseline[:, :1]), delta], dim=1)
