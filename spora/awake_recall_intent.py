"""Learned intent check for an immediate repeat of a verified memory read."""

import math
import re

import torch
from torch import nn


FORMAT = 'spora-awake-recall-intent-v1'
SUBJECT_SCOPE_CONFLICT = re.compile(
    r'更正|修正|改动前|更改前|更新以?前|原先登记|原始记录|原来的值|旧版|过去的版本|'
    r'哪[一天月日时]点?|何时|什么时候|几点|几月|'
    r'另[一外]|换.{0,4}(?:人|对象)|下[一个]位|新的对象'
)


def intent_features(tail, tokens):
    mean = torch.nn.functional.normalize(tokens.mean(dim=0), dim=0)
    return torch.cat((tail, mean)) / 2**.5


class RecallIntent(nn.Module):
    format = FORMAT

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
        if payload['format'] != cls.format or payload['fingerprint'] != fingerprint:
            raise ValueError('Recall intent head has a different format or belongs to a different decoder')
        cutoff = payload['cutoff']
        if type(cutoff) not in (int, float) or not math.isfinite(cutoff) or not 0 < cutoff <= 1:
            raise ValueError('Recall intent cutoff must be in (0, 1]')
        model = cls(width, cutoff)
        model.head.load_state_dict(payload['state'])
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise ValueError('Recall intent head parameters must be finite')
        return model.eval().requires_grad_(False)


class SameSubjectIntent(RecallIntent):
    format = 'spora-awake-same-subject-intent-v1'
