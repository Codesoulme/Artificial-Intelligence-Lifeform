"""Experimental complete-name span scoring on DeepSeek hidden states."""

import torch
from torch import nn
from torch.nn import functional as F


MAX_NAME_LENGTH = 6


class CharacterNameSpan(nn.Module):
    """Score complete character spans and a single no-address class."""

    def __init__(self, feature_width):
        super().__init__()
        self.project = nn.Linear(feature_width + 4, 48)
        self.context = nn.Conv1d(48, 48, 5, padding=2)
        self.start = nn.Linear(48, 17)
        self.end = nn.Linear(48, 17)
        self.length = nn.Parameter(torch.zeros(MAX_NAME_LENGTH))
        self.reject = nn.Linear(48, 1)

    def forward(self, features, mask):
        hidden = F.gelu(self.project(features.float())).masked_fill(~mask[..., None], 0.)
        hidden = hidden + F.gelu(self.context(hidden.transpose(1, 2)).transpose(1, 2))
        start, end = self.start(hidden), self.end(hidden)
        scores = start[..., :16] @ end[..., :16].transpose(-1, -2) / 4.
        scores = scores + start[..., 16, None] + end[..., 16].unsqueeze(-2)
        pos = torch.arange(features.shape[1], device=features.device)
        lengths = pos[None] - pos[:, None]
        scores = scores + self.length[lengths.clamp(0, MAX_NAME_LENGTH - 1)]
        allowed = (lengths >= 0) & (lengths < MAX_NAME_LENGTH) & mask[:, :, None] & mask[:, None, :]
        pooled = hidden.masked_fill(~mask[..., None], 0.).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return torch.cat((scores.masked_fill(~allowed, -torch.inf).flatten(1), self.reject(pooled)), dim=1)


def select_character_span(scores, size, margin):
    flattened = scores[:-1]
    best = int(flattened.argmax())
    start, end = divmod(best, size)
    confidence = float(flattened[best] - scores[-1])
    if confidence <= margin:
        return None, confidence
    return (start, end + 1), confidence
