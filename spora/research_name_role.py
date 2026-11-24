"""Recognize names and select the target of a memory request."""

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


class NameSpanHead(nn.Module):
    """Detect complete names before considering enrollment, including names absent from memory."""

    lengths = (2, 3, 4)

    def __init__(self, width):
        super().__init__()
        self.project = nn.Linear(width, 64, bias=False)
        self.score = nn.Sequential(nn.Linear(4 * 64, 64), nn.GELU(), nn.Linear(64, 1))
        self.length_bias = nn.Parameter(torch.zeros(len(self.lengths)))

    def forward(self, hidden):
        chars = F.gelu(self.project(F.normalize(hidden.float(), dim=-1)))
        width = chars.shape[1]
        pad = F.pad(chars, (0, 0, 1, max(self.lengths)))
        left, first = pad[:, :width], chars
        return torch.stack([self.score(torch.cat((left, first, pad[:, length:length + width],
                                                  pad[:, length + 1:length + 1 + width]), dim=-1)).squeeze(-1)
                            + bias for length, bias in zip(self.lengths, self.length_bias, strict=True)], dim=-1)


def select_nonoverlapping_spans(scores, text_length, *, threshold, limit=8):
    """Rank valid spans and keep disjoint mentions, preserving text offsets."""
    spans = [(start, start + length, float(score)) for start, options in enumerate(scores[:text_length])
             for length, score in zip(NameSpanHead.lengths, options, strict=True)
             if start + length <= text_length and score > threshold]
    picked = []
    for start, stop, _ in sorted(spans, key=lambda span: -span[2]):
        if all(stop <= a or start >= b for a, b in picked):
            picked.append((start, stop))
            if len(picked) == limit:
                break
    return sorted(picked)


def validate_enrollment(candidate, eligible):
    """Resolve only the winning exact span. A missing registration ends the request."""
    if candidate is None:
        return None
    return next((item for item in eligible if item[:2] == candidate[:2]), None)


def select_name(candidates, scores, *, threshold):
    """Allow a write only when exactly one candidate has enough evidence."""
    if len(candidates) != len(scores):
        raise ValueError('Every candidate must have a score')
    accepted = [candidate for candidate, score in zip(candidates, scores, strict=True) if score > threshold]
    return accepted[0] if len(accepted) == 1 else None
