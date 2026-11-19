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
        return cls.from_payload(torch.load(path, map_location='cpu', weights_only=True), width, fingerprint)

    @classmethod
    def from_payload(cls, payload, width, fingerprint):
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


class SelfNameIntent(RecallIntent):
    format = 'spora-awake-self-name-v1'
    other_target = re.compile(r'(?:称呼|叫)我(?:[^叫]|$)|我的(?:实际|真实|常用)?(?:名字|姓名|名称|称呼)|我一直叫什么')
    model_target = re.compile(r'底座(?:型号|名称|叫|有)|基[础座]模型|哪个模型|开发者|哪家(?:机构|公司)')

    @classmethod
    def targets_self(cls, question):
        return cls.other_target.search(question) is None and cls.model_target.search(question) is None

    @classmethod
    def load_name(cls, path, width, fingerprint, bank_spec):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        model = cls.from_payload(payload, width, fingerprint)
        bank = payload['name_bank']
        shape, dtype = bank_spec
        if (not isinstance(bank, torch.Tensor) or bank.shape != shape or bank.dtype != dtype
                or not bool(torch.isfinite(bank).all())):
            raise ValueError('Self-name matrix does not match the memory layout')
        model.register_buffer('name_bank', bank)
        return model
