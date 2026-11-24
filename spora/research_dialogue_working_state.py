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

    def encode(self, codes, features):
        valid = codes.ne(0)
        x = self.embedding(codes)
        if self.semantic is not None:
            x = x + self.semantic(F.layer_norm(features.float(), (features.shape[-1],)))
        x = (x * valid[:, :, None]).transpose(1, 2)
        for layer in self.context:
            x = (x + F.gelu(layer(x))) * valid[:, None, :]
        return x.transpose(1, 2), valid

    def boundary_scores(self, x, valid):
        spans = self.boundary(x).transpose(1, 2)
        spans = spans.masked_fill(~valid[:, None, :], -torch.inf)
        pooled = x.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        logits = torch.cat((self.absent(pooled).unsqueeze(-1), spans), dim=-1)
        return logits.reshape(x.shape[0], len(FIELDS), 2, -1)

    def forward(self, codes, features=None):
        return self.boundary_scores(*self.encode(codes, features))

    def loss(self, codes, boundaries, features=None):
        truth = boundaries.flatten()
        logits = self(codes, features).flatten(end_dim=2)
        loss = F.cross_entropy(logits, truth, reduction='none')
        return (loss * torch.where(truth > 0, 2.5, 1.)).mean()

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


class JointTurnValueWriter(TurnValueWriter):
    """Score a whole replacement span against a field's no-write decision."""

    def __init__(self, width=64, *, feature_width=0):
        super().__init__(width, feature_width=feature_width)
        self.pair_width = 8
        self.pair = nn.Linear(width, len(FIELDS) * 2 * self.pair_width)

    def span_scores(self, hidden, valid):
        boundaries = self.boundary_scores(hidden, valid)
        batch, length = valid.shape
        endpoints = self.pair(hidden).reshape(batch, length, len(FIELDS), 2, self.pair_width)
        start, end = endpoints.permute(3, 0, 2, 1, 4).unbind(0)
        interaction = (start @ end.transpose(-1, -2)) * self.pair_width ** -.5
        span_scores = boundaries[:, :, 0, 1:, None] + boundaries[:, :, 1, None, 1:] + interaction
        indices = torch.arange(length, device=hidden.device)
        widths = indices[None, :] - indices[:, None]
        span_scores = span_scores.masked_fill((widths < 0) | (widths >= SPAN_LIMIT), -torch.inf)
        absent = boundaries[..., 0].sum(dim=-1, keepdim=True)
        return absent, span_scores.flatten(start_dim=-2)

    def forward(self, codes, features=None):
        return torch.cat(self.span_scores(*self.encode(codes, features)), dim=-1)

    def loss(self, codes, boundaries, features=None):
        start, end = boundaries.unbind(-1)
        truth = torch.where(start > 0, 1 + (start - 1) * codes.shape[1] + end - 1, 0).flatten()
        logits = self(codes, features).flatten(end_dim=1)
        loss = F.cross_entropy(logits, truth, reduction='none')
        return (loss * torch.where(truth > 0, 2.5, 1.)).mean()

    @torch.inference_mode()
    def write(self, text, margin=0., *, features=None):
        logits = self(characters([text]).to(self.embedding.weight.device), features)[0]
        scores, candidates = logits[:, 1:].max(dim=-1)
        accepted = (scores - logits[:, 0] >= margin).tolist()
        spans = {}
        for field, candidate, keep in zip(FIELDS, candidates.tolist(), accepted, strict=True):
            if keep:
                start, end = divmod(candidate, len(text))
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
    """Read state token vectors by cross-attention at a decoder layer."""

    def __init__(self, hidden, width=64, *, layer_index=-1, copy_logits=True, ordered=False):
        super().__init__()
        if type(layer_index) is not int or layer_index < -1 or type(copy_logits) is not bool:
            raise ValueError('Invalid working-state read configuration')
        if type(ordered) is not bool or (ordered and not copy_logits):
            raise ValueError('Ordered reads require copy scores')
        self.layer_index = layer_index
        self.copy_logits = copy_logits
        self.query = nn.Linear(hidden, width, bias=False)
        self.key = nn.Linear(hidden, width, bias=False)
        self.value = nn.Linear(hidden, width, bias=False)
        self.fields = nn.Embedding(len(FIELDS), width)
        self.positions = nn.Embedding(SPAN_LIMIT, width)
        self.output = nn.Linear(width, hidden, bias=False)
        self.copy_scale = nn.Parameter(torch.zeros(()))
        self.order_scale = nn.Parameter(torch.zeros(())) if ordered else None
        nn.init.zeros_(self.output.weight)

    def attention(self, hidden, embeddings, owners, positions, continuation=None, *, eligible=None):
        query = self.query(hidden.float())
        keys = self.key(embeddings.float()) + self.fields(owners) + self.positions(positions)
        weight = (query @ keys.T) * (keys.shape[-1] ** -.5)
        if continuation is not None:
            weight = weight + 8. * self.order_scale.tanh() * continuation
        if eligible is not None:
            weight = weight.masked_fill(~eligible, torch.finfo(weight.dtype).min)
            return weight.softmax(dim=-1).masked_fill(~eligible, 0.)
        return weight.softmax(dim=-1)

    def forward(self, hidden, embeddings, owners, positions):
        weights = self.attention(hidden, embeddings, owners, positions)
        values = self.value(embeddings.float()) + self.fields(owners)
        update = self.output(weights @ values)
        return hidden + update.to(hidden.dtype)

    @contextmanager
    def active(self, decoder, state, *, targets=None, prompt_tokens=None):
        if decoder._memory_context.get() is not None or _READ_SCOPE.get() is not None:
            raise ValueError('Working condition reads require ordinary dialogue')
        ids, owners, positions = state.tokens(decoder.base.device)
        sizes = tuple(len(value) for value in state.slots if value)
        supervision = []
        if not ids.numel():
            yield supervision
            return
        with torch.no_grad():
            embeddings = decoder.base.get_input_embeddings()(ids)
        marker = object()
        sequence, projected, inferred_prompt = None, None, None

        def capture(_base, args, kwargs):
            nonlocal sequence, projected, inferred_prompt
            if _READ_SCOPE.get() is not marker or decoder._memory_context.get() is not None:
                return
            tokens = kwargs.get('input_ids', args[0] if args else None)
            if tokens is None:
                raise ValueError('Ordered working-state reads require token input')
            past = kwargs.get('past_key_values')
            continuing = past is not None and past.get_seq_length() > 0
            if continuing and sequence is None:
                raise ValueError('Ordered working-state reads require the current prompt')
            sequence = torch.cat((sequence, tokens), dim=1) if continuing else tokens
            if not continuing:
                inferred_prompt = tokens.shape[1]
            keep = kwargs.get('logits_to_keep', 0)
            offset = sequence.shape[1] - tokens.shape[1]
            projected = (keep + offset if isinstance(keep, torch.Tensor) else
                         torch.arange(sequence.shape[1] - (keep or tokens.shape[1]), sequence.shape[1],
                                      device=tokens.device))

        def read(_layer, _inputs, output):
            if _READ_SCOPE.get() is not marker or decoder._memory_context.get() is not None:
                return output
            if isinstance(output, tuple):
                return (self(output[0], embeddings, owners, positions), *output[1:])
            return self(output, embeddings, owners, positions)

        def copy(_head, inputs, output):
            if _READ_SCOPE.get() is not marker or decoder._memory_context.get() is not None:
                return output
            prefix = inferred_prompt if prompt_tokens is None else prompt_tokens
            weights = self.copy_attention(
                inputs[0], embeddings, owners, positions, ids, sequence, projected, prefix, sizes,
            )
            if targets is not None:
                matching = ids[None, :] == targets[:, None]
                selected = matching.any(dim=-1)
                if bool(selected.any()):
                    mass = (weights[0] * matching).sum(dim=-1)
                    supervision.append(-mass[selected].clamp_min(1e-8).log().mean())
            if self.copy_logits:
                return self.copy_scores(inputs[0], output, ids, weights)
            return output

        with ExitStack() as cleanup:
            cleanup.callback(_READ_SCOPE.reset, _READ_SCOPE.set(marker))
            if self.order_scale is not None:
                input_handle = decoder.base.register_forward_pre_hook(capture, with_kwargs=True)
                cleanup.callback(input_handle.remove)
            layer_handle = decoder.base.model.layers[self.layer_index].register_forward_hook(read)
            cleanup.callback(layer_handle.remove)
            if targets is not None or self.copy_logits:
                head_handle = decoder.base.lm_head.register_forward_hook(copy)
                cleanup.callback(head_handle.remove)
            yield supervision

    def copy_attention(self, hidden, embeddings, owners, positions, ids, sequence, projected, prefix, sizes):
        continuation = (None if self.order_scale is None else
                        slot_continuations(sequence, projected, prefix, ids, positions))
        return self.attention(hidden, embeddings, owners, positions, continuation)

    def copy_scores(self, hidden, logits, ids, weights):
        bonus = (12. * self.copy_scale.tanh() * weights).to(logits.dtype)
        return logits.scatter_add(-1, ids.expand_as(bonus), bonus)


class MixtureWorkingTurnReader(WorkingTurnReader):
    """Mix the language vocabulary with a gated distribution over internal state tokens."""

    def __init__(self, hidden, width=64):
        super().__init__(hidden, width, ordered=True)
        self.copy_gate = nn.Linear(hidden, 1)
        nn.init.zeros_(self.copy_gate.weight)
        nn.init.constant_(self.copy_gate.bias, -4.)
        self.copy_scale.requires_grad_(False)

    def copy_scores(self, hidden, logits, ids, weights):
        normalized = F.rms_norm(hidden.float(), (hidden.shape[-1],))
        gate = self.copy_gate(normalized)
        has_state_mass = weights.sum(dim=-1, keepdim=True).gt(0)
        gate = gate.masked_fill(~has_state_mass, torch.finfo(gate.dtype).min)
        language = logits.float().log_softmax(dim=-1) + F.logsigmoid(-gate)
        unique, inverse = ids.unique(return_inverse=True)
        state_mass = weights.new_zeros((*weights.shape[:-1], unique.numel()))
        state_mass = state_mass.scatter_add(-1, inverse.expand_as(weights), weights)
        selected = unique.expand_as(state_mass)
        copied = state_mass.clamp_min(torch.finfo(state_mass.dtype).tiny).log() + F.logsigmoid(gate)
        combined = torch.logaddexp(language.gather(-1, selected), copied)
        return language.scatter(-1, selected, combined)


class CoveredWorkingTurnReader(MixtureWorkingTurnReader):
    """Use reply coverage or span continuation to refine working-state attention."""

    def __init__(self, hidden, width=64, *, follow_span=False, suppress_completed=True):
        super().__init__(hidden, width)
        self.follow_span = follow_span
        self.suppress_completed = suppress_completed

    def copy_attention(self, hidden, embeddings, owners, positions, ids, sequence, projected, prefix, sizes):
        continuation = slot_continuations(sequence, projected, prefix, ids, positions)
        eligible = (~slot_coverage(sequence, projected, prefix, ids, sizes) if self.suppress_completed
                    else torch.ones_like(continuation))
        if self.follow_span:
            ongoing = continuation & eligible
            eligible = torch.where(ongoing.any(dim=-1, keepdim=True), ongoing, eligible)
        return self.attention(hidden, embeddings, owners, positions, continuation, eligible=eligible)


def slot_coverage(sequence, projected, prompt_tokens, ids, sizes):
    """Mark fields already emitted as exact spans at each projected reply position."""
    reply = sequence[:, prompt_tokens:]
    fields = []
    for value in ids.split(sizes):
        span = value.numel()
        if reply.shape[-1] < span:
            covered = torch.zeros((reply.shape[0], projected.numel()), dtype=torch.bool, device=reply.device)
        else:
            ends = reply.unfold(-1, span, 1).eq(value).all(dim=-1)
            seen = ends.cumsum(dim=-1).gt(0)
            index = projected - prompt_tokens - span + 1
            covered = seen[:, index.clamp_min(0)] & index.ge(0)
        fields.append(covered[..., None].expand(-1, -1, span))
    return torch.cat(fields, dim=-1)


def slot_continuations(sequence, projected, prompt_tokens, ids, positions):
    """Mark next tokens of the longest slot prefix at the end of the reply so far."""
    lags = torch.arange(SPAN_LIMIT - 1, device=ids.device)
    recent = projected[:, None] - lags
    suffixes = sequence[:, recent.clamp_min(0)].masked_fill(recent < prompt_tokens, -1)
    precedes = torch.arange(ids.numel(), device=ids.device)[:, None] - 1 - lags
    prefixes = ids[precedes.clamp_min(0)]
    needed = lags < positions[:, None]
    matches = ((suffixes[:, :, None] == prefixes) | ~needed).all(dim=-1)
    lengths = torch.where(matches, positions, 0)
    longest = lengths.amax(dim=-1, keepdim=True)
    return (lengths == longest) & longest.gt(0)
