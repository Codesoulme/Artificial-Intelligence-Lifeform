"""Learned intent check for an immediate repeat of a verified memory read."""

import math

import torch
from torch import nn


FORMAT = 'spora-awake-recall-intent-v1'


def intent_features(tail, tokens):
    mean = torch.nn.functional.normalize(tokens.mean(dim=0), dim=0)
    return torch.cat((tail, mean)) / 2**.5


class RecallIntent(nn.Module):
    def __init__(self, width, cutoff):
        super().__init__()
        self.head = nn.Linear(2 * width, 1)
        self.cutoff = cutoff

    def forward(self, tail, tokens):
        score = self.head(intent_features(tail, tokens)).sigmoid().item()
        return score >= self.cutoff

    @classmethod
    def load(cls, path, width, fingerprint):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['format'] != FORMAT or payload['fingerprint'] != fingerprint:
            raise ValueError('Recall intent head belongs to a different decoder')
        cutoff = payload['cutoff']
        if type(cutoff) not in (int, float) or not math.isfinite(cutoff) or not 0 < cutoff <= 1:
            raise ValueError('Recall intent cutoff must be in (0, 1]')
        model = cls(width, cutoff)
        model.head.load_state_dict(payload['state'])
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise ValueError('Recall intent head parameters must be finite')
        return model.eval().requires_grad_(False)
