"""Score a registered mention as the target of a memory request."""

import torch
from torch import nn
from torch.nn import functional as F


def mention_features(hidden, start, stop):
    """Read the candidate, two right boundary positions, and the complete request."""
    if not 0 <= start < stop <= len(hidden):
        raise ValueError('Invalid mention offsets')
    zero = hidden.new_zeros(hidden.shape[-1])
    left = hidden[start - 1] if start else zero
    first = hidden[stop] if stop < len(hidden) else zero
    second = hidden[stop + 1] if stop + 1 < len(hidden) else zero
    parts = (hidden[start:stop].mean(0), left, first, second, hidden.mean(0), hidden[-1])
    return F.normalize(torch.stack(parts).float(), dim=-1)


class NameRoleHead(nn.Module):
    def __init__(self, width, *, contextual):
        super().__init__()
        self.parts = 6 if contextual else 4
        self.layers = nn.Sequential(nn.Linear(self.parts * width, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, features):
        return self.layers(features[..., :self.parts, :].flatten(-2)).squeeze(-1)


def select_name(candidates, scores, *, threshold):
    """Allow a write only when exactly one candidate has enough evidence."""
    if len(candidates) != len(scores):
        raise ValueError('Every candidate must have a score')
    accepted = [candidate for candidate, score in zip(candidates, scores, strict=True) if score > threshold]
    return accepted[0] if len(accepted) == 1 else None
