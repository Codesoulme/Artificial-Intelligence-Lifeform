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


class JointNameRoleHead(nn.Module):
    """Score complete names and select a request target against an abstention logit."""

    def __init__(self, width):
        super().__init__()
        self.boundary = NameRoleHead(width, contextual=False)
        self.project = nn.Linear(6 * width, 64)
        self.role = nn.Linear(128, 1)
        self.abstain = nn.Linear(64, 1)

    def forward(self, features, present):
        boundary = self.boundary(features)
        encoded = F.gelu(self.project(features.flatten(-2)))
        pooled = encoded.masked_fill(~present[..., None], 0).sum(dim=1) / present.sum(dim=1, keepdim=True).clamp_min(1)
        context = pooled.unsqueeze(1).expand_as(encoded)
        role = self.role(torch.cat((encoded, context), dim=-1)).squeeze(-1)
        candidate = (role + F.logsigmoid(boundary)).masked_fill(~present, -torch.inf)
        decision = torch.cat((candidate, self.abstain(pooled)), dim=-1)
        return decision, boundary


def compare_name_scores(decision, boundary, present):
    """Return the best real mention and its weakest decision margin."""
    candidates = decision[..., :-1].masked_fill(~present, -torch.inf)
    best, index = candidates.max(-1)
    other = candidates.scatter(-1, index[:, None], -torch.inf).amax(-1)
    margin = best - torch.maximum(other, decision[..., -1])
    completeness = boundary.gather(-1, index[:, None]).squeeze(-1)
    return index, torch.minimum(margin, completeness)


def select_name(candidates, scores, *, threshold):
    """Allow a write only when exactly one candidate has enough evidence."""
    if len(candidates) != len(scores):
        raise ValueError('Every candidate must have a score')
    accepted = [candidate for candidate, score in zip(candidates, scores, strict=True) if score > threshold]
    return accepted[0] if len(accepted) == 1 else None
