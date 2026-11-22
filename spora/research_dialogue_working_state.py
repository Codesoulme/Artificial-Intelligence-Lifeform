"""A learned, session-local state for explicit conditions in user turns."""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


FIELDS = ('day', 'time', 'place', 'gear', 'person', 'quantity', 'direction')
CHAR_BUCKETS = 8192
SPAN_LIMIT = 14
TURN_LIMIT = 160
_READ_SCOPE = ContextVar('spora_working_turn_read', default=None)


def characters(texts):
    if not texts or any(not isinstance(text, str) or not 0 < len(text) <= TURN_LIMIT for text in texts):
        raise ValueError('Working-state turns must be nonempty and fit the character limit')
    width = max(map(len, texts))
    return torch.tensor([[1 + ord(char) % (CHAR_BUCKETS - 1) for char in text]
                         + [0] * (width - len(text)) for text in texts], dtype=torch.long)


class TurnValueWriter(nn.Module):
    """Predict a current value span or an explicit no-write address per field."""

    def __init__(self, width=64, *, feature_width=0):
        super().__init__()
        self.embedding = nn.Embedding(CHAR_BUCKETS, width, padding_idx=0)
        self.semantic = nn.Linear(feature_width, width, bias=False) if feature_width else None
        self.context = nn.ModuleList(nn.Conv1d(width, width, 5, dilation=d, padding=2 * d)
                                     for d in (1, 2, 4))
        self.boundary = nn.Linear(width, len(FIELDS) * 2)
        self.absent = nn.Linear(width, len(FIELDS) * 2)

    def forward(self, codes, features=None):
        valid = codes.ne(0)
        x = self.embedding(codes)
        if self.semantic is not None:
            x = x + self.semantic(F.layer_norm(features.float(), (features.shape[-1],)))
        x = (x * valid[:, :, None]).transpose(1, 2)
        for layer in self.context:
            x = (x + F.gelu(layer(x))) * valid[:, None, :]
        x = x.transpose(1, 2)
        spans = self.boundary(x).transpose(1, 2)
        spans = spans.masked_fill(~valid[:, None, :], -torch.inf)
        pooled = x.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        logits = torch.cat((self.absent(pooled).unsqueeze(-1), spans), dim=-1)
        return logits.reshape(codes.shape[0], len(FIELDS), 2, -1)

    @torch.inference_mode()
    def write(self, text, margin=2., *, features=None):
        logits = self(characters([text]).to(self.embedding.weight.device), features)[0].cpu()
        length = len(text)
        indices = torch.arange(length)
        width = indices[None, :] - indices[:, None]
        possible = (width >= 0) & (width < SPAN_LIMIT)
        spans = {}
        for field, probs in zip(FIELDS, logits, strict=True):
            candidates = probs[0, 1:, None] + probs[1, None, 1:]
            candidates = candidates.masked_fill(~possible, -torch.inf)
            best = int(candidates.argmax())
            start, end = divmod(best, length)
            if float(candidates[start, end] - probs[:, 0].sum()) >= margin:
                spans[field] = text[start:end + 1]
        return spans


@dataclass(frozen=True)
class WorkingTurnState:
    """Store bounded input-token sequences by model address, for one conversation."""

    slots: tuple[tuple[int, ...], ...] = ((),) * len(FIELDS)

    def update(self, changes):
        if changes.keys() - set(FIELDS):
            raise ValueError('Unknown working-state address')
        if any(not isinstance(value, tuple) or not 1 <= len(value) <= SPAN_LIMIT
               or any(type(token) is not int or token < 0 for token in value) for value in changes.values()):
            raise ValueError('Working-state values must be bounded token sequences')
        return WorkingTurnState(tuple(changes.get(name, old) for name, old in zip(FIELDS, self.slots, strict=True)))

    def tokens(self, device):
        ids, owners, positions = [], [], []
        for index, value in enumerate(self.slots):
            ids.extend(value)
            owners.extend([index] * len(value))
            positions.extend(range(len(value)))
        return (torch.tensor(ids, dtype=torch.long, device=device),
                torch.tensor(owners, dtype=torch.long, device=device),
                torch.tensor(positions, dtype=torch.long, device=device))


class WorkingTurnReader(nn.Module):
    """Read state token vectors by cross-attention on the final decoder layer."""

    def __init__(self, hidden, width=64):
        super().__init__()
        self.query = nn.Linear(hidden, width, bias=False)
        self.key = nn.Linear(hidden, width, bias=False)
        self.value = nn.Linear(hidden, width, bias=False)
        self.fields = nn.Embedding(len(FIELDS), width)
        self.positions = nn.Embedding(SPAN_LIMIT, width)
        self.output = nn.Linear(width, hidden, bias=False)
        self.copy_scale = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.output.weight)

    def attention(self, hidden, embeddings, owners, positions):
        query = self.query(hidden.float())
        keys = self.key(embeddings.float()) + self.fields(owners) + self.positions(positions)
        weight = (query @ keys.T) * (keys.shape[-1] ** -.5)
        return weight.softmax(dim=-1)

    def forward(self, hidden, embeddings, owners, positions):
        weights = self.attention(hidden, embeddings, owners, positions)
        values = self.value(embeddings.float()) + self.fields(owners)
        update = self.output(weights @ values)
        return hidden + update.to(hidden.dtype)

    @contextmanager
    def active(self, decoder, state, *, targets=None):
        if decoder._memory_context.get() is not None or _READ_SCOPE.get() is not None:
            raise ValueError('Working condition reads require ordinary dialogue')
        ids, owners, positions = state.tokens(decoder.base.device)
        supervision = []
        if not ids.numel():
            yield supervision
            return
        with torch.no_grad():
            embeddings = decoder.base.get_input_embeddings()(ids)
        marker = object()

        def read(_layer, _inputs, output):
            if _READ_SCOPE.get() is not marker or decoder._memory_context.get() is not None:
                return output
            if isinstance(output, tuple):
                return (self(output[0], embeddings, owners, positions), *output[1:])
            return self(output, embeddings, owners, positions)

        def copy(_head, inputs, output):
            if _READ_SCOPE.get() is not marker or decoder._memory_context.get() is not None:
                return output
            weights = self.attention(inputs[0], embeddings, owners, positions)
            if targets is not None:
                matching = ids[None, :] == targets[:, None]
                selected = matching.any(dim=-1)
                if bool(selected.any()):
                    mass = (weights[0] * matching).sum(dim=-1)
                    supervision.append(-mass[selected].clamp_min(1e-8).log().mean())
            bonus = (12. * self.copy_scale.tanh() * weights).to(output.dtype)
            return output.scatter_add(-1, ids.expand_as(bonus), bonus)

        with ExitStack() as cleanup:
            cleanup.callback(_READ_SCOPE.reset, _READ_SCOPE.set(marker))
            layer_handle = decoder.base.model.layers[-1].register_forward_hook(read)
            cleanup.callback(layer_handle.remove)
            head_handle = decoder.base.lm_head.register_forward_hook(copy)
            cleanup.callback(head_handle.remove)
            yield supervision
