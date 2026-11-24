"""Distinguish a field update from selection of the complete replacement value."""

import torch
from torch import nn
from torch.nn import functional as F

from .research_dialogue_working_state import FIELDS, JointTurnValueWriter, characters


class SeparateTurnValueWriter(JointTurnValueWriter):
    def __init__(self, width=64, *, feature_width=0):
        super().__init__(width, feature_width=feature_width)
        self.operation = nn.Linear(width, len(FIELDS))
        self.activity = nn.Linear(width, len(FIELDS))

    def scores(self, codes, features=None):
        hidden, valid = self.encode(codes, features)
        attention = self.operation(hidden).masked_fill(~valid[..., None], -torch.inf).softmax(dim=1)
        witnesses = (attention * self.activity(hidden)).sum(dim=1)
        _, spans = self.span_scores(hidden, valid)
        return witnesses, spans

    def loss(self, codes, boundaries, features=None):
        operation, spans = self.scores(codes, features)
        start, end = boundaries.unbind(-1)
        writes = start.gt(0)
        gate_loss = F.binary_cross_entropy_with_logits(operation, writes.float(), reduction='none')
        gate_loss = (gate_loss * torch.where(writes, 2.5, 1.)).mean()
        if not bool(writes.any()):
            return gate_loss
        indexed = ((start[writes] - 1) * codes.shape[1] + end[writes] - 1)
        candidates = spans[writes]
        span_loss = F.cross_entropy(candidates, indexed)
        truth = candidates.gather(1, indexed[:, None]).squeeze(1)
        rivals = candidates.scatter(1, indexed[:, None], -torch.inf).amax(dim=1)
        # Compare against the strongest competing interval, including nested
        # and across-field spans whenever they outrank the full new value.
        ranking = F.softplus(1. + rivals - truth).mean()
        return gate_loss + span_loss + .25 * ranking

    @torch.inference_mode()
    def write(self, text, margin=0., *, features=None):
        codes = characters([text]).to(self.activity.weight.device)
        operation, spans = self.scores(codes, features)
        selected = spans[0].argmax(dim=-1).tolist()
        accepted = operation[0].ge(margin).tolist()
        changes = {}
        for field, flat, active in zip(FIELDS, selected, accepted, strict=True):
            if active:
                i, j = divmod(flat, len(text))
                changes[field] = text[i:j + 1]
        return changes
