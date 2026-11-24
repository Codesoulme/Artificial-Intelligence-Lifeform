"""Learn an output action that reads one whole field from the working state."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .research_dialogue_working_state import FIELDS


class SpanActionHead(nn.Module):
    """Score ordinary vocabulary actions and whole-field reads in one distribution."""

    def __init__(self, hidden, width=64):
        super().__init__()
        self.query = nn.Linear(hidden, width, bias=False)
        self.key = nn.Linear(hidden, width, bias=False)
        self.field = nn.Embedding(len(FIELDS), width)
        self.context = nn.Linear(width, hidden, bias=False)
        self.language = nn.Sequential(nn.Linear(hidden, width, bias=False), nn.Tanh(),
                                      nn.Linear(width, hidden, bias=False))
        self.mode = nn.Linear(hidden, 1)
        self.route = nn.Linear(hidden, 1)
        nn.init.zeros_(self.context.weight)
        nn.init.zeros_(self.language[-1].weight)

    def field_keys(self, state, embeddings):
        keys = self.field.weight.clone()
        for field, span in enumerate(state.slots):
            if span:
                ids = torch.tensor(span, device=keys.device)
                vectors = embeddings(ids).float()
                keys[field] = keys[field] + self.key(F.rms_norm(vectors, (vectors.shape[-1],))).mean(dim=0)
        return keys

    def route_logit(self, hidden):
        return self.route(hidden.float()).squeeze(-1)

    def forward(self, hidden, keys, available, vocabulary):
        hidden_fp32 = hidden.float()
        pointer = (self.query(hidden_fp32) @ keys.T) * keys.shape[-1] ** -.5
        pointer = pointer.masked_fill(~available, torch.finfo(pointer.dtype).min)
        attention = pointer.softmax(dim=-1).masked_fill(~available, 0.)
        conditioned = hidden_fp32 + self.context(attention @ keys)
        word_hidden = conditioned + self.language(conditioned)
        words = vocabulary(word_hidden.to(hidden.dtype)).float().log_softmax(dim=-1)
        gate = self.mode(conditioned)
        gate = gate.masked_fill(~available.any(dim=-1, keepdim=True), torch.finfo(gate.dtype).min)
        words = words + F.logsigmoid(-gate)
        spans = (pointer.log_softmax(dim=-1) + F.logsigmoid(gate)).masked_fill(~available, -torch.inf)
        return torch.cat((words, spans), dim=-1)


@dataclass(frozen=True)
class ActionTarget:
    """One prediction position and the action to supervise at that position."""

    offset: int
    action: int


def compile_actions(segments, state, vocabulary_size, eos_token):
    """Align action labels with the token sequence consumed by the backbone."""
    tokens, targets, completed = [], [], set()
    for field, sequence in segments:
        if field is None:
            for token in sequence:
                targets.append(ActionTarget(len(tokens), token))
                tokens.append(token)
        else:
            if field in completed or tuple(sequence) != state.slots[field] or not sequence:
                raise ValueError('A copied field must name one unused, nonempty state span')
            targets.append(ActionTarget(len(tokens), vocabulary_size + field))
            tokens.extend(sequence)
            completed.add(field)
    targets.append(ActionTarget(len(tokens), eos_token))
    return tokens, targets


def available_fields(state, targets, vocabulary_size, device):
    """Mark each field available up to its annotated read action."""
    steps = torch.arange(len(targets), device=device)[:, None]
    used_at = [len(targets) if value else -1 for value in state.slots]
    for index, target in enumerate(targets):
        if target.action >= vocabulary_size:
            used_at[target.action - vocabulary_size] = index
    return steps <= torch.tensor(used_at, device=device)


def emit_action(action, state, remaining, budget, vocabulary_size):
    """Expand a word or one selected state field and consume that field."""
    if action < vocabulary_size:
        return (action,)
    field = action - vocabulary_size
    span = state.slots[field]
    if field not in remaining or not 0 < len(span) <= budget:
        raise ValueError('The decoder selected an unavailable field')
    remaining.remove(field)
    return span
