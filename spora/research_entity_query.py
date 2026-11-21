"""Separate full person mentions from selection of a memory query target."""

import torch
from torch import nn
from torch.nn import functional as F


class MentionTagger(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.project = nn.Linear(width, 64)
        self.context = nn.Conv1d(64, 64, 5, padding=2)
        self.tags = nn.Linear(64, 3)

    def forward(self, features, mask):
        hidden = F.gelu(self.project(features.float())).masked_fill(~mask[..., None], 0.)
        return self.tags(hidden + F.gelu(self.context(hidden.transpose(1, 2)).transpose(1, 2)))


def mention_spans(tags):
    """Decode O/B/I, ignoring an I tag without a preceding start."""
    spans = []
    start = None
    for end, tag in enumerate([*tags, 0]):
        if tag == 2:
            continue
        if start is not None:
            spans.append((start, end))
        start = end if tag == 1 else None
    return spans


class QueryTargetSelector(nn.Module):
    """Rank complete detected mentions against the no-single-target option."""

    def __init__(self, width):
        super().__init__()
        self.project = nn.Linear(width, 64)
        self.context = nn.Conv1d(64, 64, 9, padding=4)
        self.target = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))
        self.reject = nn.Linear(64, 1)

    def forward(self, features, valid_chars, candidate_chars):
        hidden = F.gelu(self.project(features.float())).masked_fill(~valid_chars[..., None], 0.)
        hidden = hidden + F.gelu(self.context(hidden.transpose(1, 2)).transpose(1, 2))
        query = hidden.masked_fill(~valid_chars[..., None], 0.).sum(1) / valid_chars.sum(1, keepdim=True)
        lengths = candidate_chars.sum(-1)
        mentions = candidate_chars.float() @ hidden / lengths[..., None].clamp_min(1)
        context = query[:, None, :].expand_as(mentions)
        scores = self.target(torch.cat((mentions, context), dim=-1)).squeeze(-1)
        scores = scores.masked_fill(lengths == 0, -torch.inf)
        return torch.cat((scores, self.reject(query)), dim=-1)
