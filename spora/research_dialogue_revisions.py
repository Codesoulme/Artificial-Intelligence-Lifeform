"""Learn to update temporary dialogue requirements from the current user turn."""

from enum import IntEnum

import torch
from torch import nn
from torch.nn import functional as F

from .research_dialogue_working_state import SPAN_LIMIT, TurnValueWriter


class Revision(IntEnum):
    KEEP = 0
    SET = 1
    REMOVE = 2


def revise(state, changes):
    next_state = dict(state)
    for field, value in changes.items():
        if value is None:
            next_state.pop(field, None)
        else:
            next_state[field] = value
    return next_state


class DialogueRevisionWriter(TurnValueWriter):
    """Separate a field's operation from the phrase it should copy on a write."""

    def __init__(self, feature_width, fields, width=48):
        super().__init__(width, feature_width=feature_width, fields=fields)
        self.operation = nn.Linear(width, len(self.fields) * len(Revision))

    def forward(self, codes, features):
        hidden, valid = self.encode(codes, features)
        pooled = hidden.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        operations = self.operation(pooled).reshape(len(codes), len(self.fields), len(Revision))
        return operations, self.boundary_scores(hidden, valid)

    def loss(self, codes, operations, boundaries, features):
        actions, spans = self(codes, features)
        choice_loss = F.cross_entropy(actions.flatten(0, 1), operations.flatten(),
                                      weight=actions.new_tensor((1., 3., 5.)))
        write = operations == Revision.SET
        # Only the addressed fields with a new value have a span target.
        copy_loss = F.cross_entropy(spans[write].flatten(0, 1), boundaries[write].flatten(), reduction='sum')
        return choice_loss + copy_loss / (2 * write.sum().clamp_min(1))

    @torch.inference_mode()
    def updates(self, texts, codes, features):
        operations, spans = self(codes, features)
        decisions = operations.argmax(-1).cpu().tolist()
        indices = torch.arange(codes.shape[1], device=codes.device)
        width = indices[None, :] - indices[:, None]
        possible = (width >= 0) & (width < SPAN_LIMIT)
        candidates = (spans[:, :, 0, 1:, None] + spans[:, :, 1, None, 1:]).masked_fill(~possible, -torch.inf)
        spans = candidates.flatten(2).argmax(-1).cpu().tolist()
        output = []
        for text, acts, ranges in zip(texts, decisions, spans, strict=True):
            changes = {}
            for field, action, location in zip(self.fields, acts, ranges, strict=True):
                if action == Revision.REMOVE:
                    changes[field] = None
                elif action == Revision.SET:
                    start, end = divmod(location, codes.shape[1])
                    changes[field] = text[start:end + 1]
            output.append(changes)
        return output
