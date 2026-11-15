"""Research-only semantic slot bridge for an otherwise frozen decoder."""
from contextlib import contextmanager
from contextvars import ContextVar
import math

import torch
from torch import nn


_ACTIVE_SLOT = ContextVar('spora_semantic_slot', default=None)
_ACTIVE_SEQUENCE = ContextVar('spora_semantic_slot_sequence', default=None)


def _consume_slot():
    state = _ACTIVE_SLOT.get()
    if state is None or state['used']:
        return None
    state['used'] = True
    return state['slot']


def _consume_sequence_slot():
    state = _ACTIVE_SEQUENCE.get()
    if state is None or state['index'] >= len(state['slots']):
        return None
    slot = state['slots'][state['index']]
    state['index'] += 1
    return slot


class SemanticSlotBridge(nn.Module):
    """Low-rank map from a persistent semantic slot to hidden residual space."""

    def __init__(self, width, rank=16):
        super().__init__()
        if type(width) is not int or width < 2 or type(rank) is not int or not 1 <= rank <= width:
            raise ValueError('Invalid semantic slot dimensions')
        self.down = nn.Linear(width, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, width, bias=False, dtype=torch.float32)
        nn.init.normal_(self.down.weight, std=width ** -0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, slot):
        if not isinstance(slot, torch.Tensor) or slot.shape[-1] != self.down.in_features:
            raise ValueError('Semantic slot has an incompatible width')
        return self.up(torch.tanh(self.down(slot.float())))


class GainSemanticSlotBridge(SemanticSlotBridge):
    """Semantic bridge with a learnable identity path and low-rank correction."""

    def __init__(self, width, rank=16, initial_gain=32.0):
        if not isinstance(initial_gain, (int, float)) or initial_gain <= 0:
            raise ValueError('Initial semantic gain must be positive')
        super().__init__(width, rank=rank)
        self.log_gain = nn.Parameter(torch.tensor(math.log(float(initial_gain))))

    def forward(self, slot):
        return self.log_gain.exp() * slot.float() + super().forward(slot)


class SemanticSlotLMHead(nn.Module):
    """Apply a slot residual before the frozen native language head."""

    def __init__(self, base, bridge):
        super().__init__()
        if not isinstance(base, nn.Linear) or not isinstance(bridge, SemanticSlotBridge):
            raise ValueError('A linear language head and semantic bridge are required')
        self.base = base
        self.bridge = bridge

    def forward(self, hidden):
        slot = _consume_sequence_slot()
        if slot is None:
            slot = _consume_slot()
        if slot is None:
            return self.base(hidden)
        residual = self.bridge(slot).to(device=hidden.device, dtype=hidden.dtype)
        while residual.ndim < hidden.ndim:
            residual = residual.unsqueeze(0)
        return self.base(hidden + residual)


class ScaledSemanticSlotLMHead(nn.Module):
    """Capacity-probe head that adds the stored semantic vector directly."""

    def __init__(self, base, scale):
        super().__init__()
        if not isinstance(base, nn.Linear) or not isinstance(scale, (int, float)) or scale <= 0:
            raise ValueError('A linear head and positive scale are required')
        self.base = base
        self.scale = float(scale)

    def forward(self, hidden):
        slot = _consume_sequence_slot()
        if slot is None:
            slot = _consume_slot()
        if slot is None:
            return self.base(hidden)
        residual = (self.scale * slot).to(device=hidden.device, dtype=hidden.dtype)
        while residual.ndim < hidden.ndim:
            residual = residual.unsqueeze(0)
        return self.base(hidden + residual)


def install(model, *, rank=16):
    """Install a zero-output bridge and return its trainable module."""
    head = getattr(model, 'lm_head', None)
    if isinstance(head, SemanticSlotLMHead) or not isinstance(head, nn.Linear):
        raise ValueError('Expected an unwrapped linear language head')
    bridge = SemanticSlotBridge(head.in_features, rank=rank).to(device=head.weight.device)
    model.lm_head = SemanticSlotLMHead(head, bridge)
    return bridge


def install_scaled(model, *, scale=32.0):
    """Install the direct semantic capacity-probe head."""
    head = getattr(model, 'lm_head', None)
    if isinstance(head, (SemanticSlotLMHead, ScaledSemanticSlotLMHead)) or not isinstance(head, nn.Linear):
        raise ValueError('Expected an unwrapped linear language head')
    model.lm_head = ScaledSemanticSlotLMHead(head, scale)
    return model.lm_head


def install_gain(model, *, rank=16, initial_gain=32.0):
    """Install a trainable identity-plus-low-rank semantic bridge."""
    head = getattr(model, 'lm_head', None)
    if isinstance(head, (SemanticSlotLMHead, ScaledSemanticSlotLMHead)) or not isinstance(head, nn.Linear):
        raise ValueError('Expected an unwrapped linear language head')
    bridge = GainSemanticSlotBridge(head.in_features, rank=rank,
                                    initial_gain=initial_gain).to(device=head.weight.device)
    model.lm_head = SemanticSlotLMHead(head, bridge)
    return bridge


class SequenceSemanticSlotLMHead(nn.Module):
    """Apply one semantic vector per cached decoder step."""

    def __init__(self, base, scale=32.0):
        super().__init__()
        if not isinstance(base, nn.Linear) or not isinstance(scale, (int, float)) or scale <= 0:
            raise ValueError('A linear head and positive scale are required')
        self.base = base
        self.scale = float(scale)

    def forward(self, hidden):
        slot = _consume_sequence_slot()
        if slot is None:
            slot = _consume_slot()
        if slot is None:
            return self.base(hidden)
        residual = (self.scale * slot).to(device=hidden.device, dtype=hidden.dtype)
        while residual.ndim < hidden.ndim:
            residual = residual.unsqueeze(0)
        return self.base(hidden + residual)


def install_sequence_scaled(model, *, scale=32.0):
    """Install a direct sequence-slot capacity probe head."""
    head = getattr(model, 'lm_head', None)
    if isinstance(head, (SemanticSlotLMHead, ScaledSemanticSlotLMHead,
                         SequenceSemanticSlotLMHead)) or not isinstance(head, nn.Linear):
        raise ValueError('Expected an unwrapped linear language head')
    model.lm_head = SequenceSemanticSlotLMHead(head, scale=scale)
    return model.lm_head


@contextmanager
def use_slot(slot):
    """Expose one isolated slot for a call, including cached generation steps."""
    if not isinstance(slot, torch.Tensor) or slot.ndim not in (1, 2):
        raise ValueError('Semantic slot must be a vector or a batch of vectors')
    if not slot.is_floating_point() or not bool(torch.isfinite(slot).all()):
        raise ValueError('Semantic slot must be finite and floating point')
    token = _ACTIVE_SLOT.set({'slot': slot.detach(), 'used': False})
    try:
        yield
    finally:
        _ACTIVE_SLOT.reset(token)


@contextmanager
def use_slot_sequence(slots):
    """Expose a finite sequence of semantic vectors for cached generation."""
    if isinstance(slots, torch.Tensor):
        if slots.ndim != 2:
            raise ValueError('Semantic slot sequence must be rank two')
        slots = tuple(slots[index] for index in range(slots.shape[0]))
    if not isinstance(slots, (tuple, list)) or not slots:
        raise ValueError('A nonempty semantic slot sequence is required')
    normalized = []
    for slot in slots:
        if not isinstance(slot, torch.Tensor) or slot.ndim != 1 or not slot.is_floating_point():
            raise ValueError('Every semantic sequence slot must be a floating vector')
        if not bool(torch.isfinite(slot).all()):
            raise ValueError('Semantic sequence slots must be finite')
        normalized.append(slot.detach())
    token = _ACTIVE_SEQUENCE.set({'slots': tuple(normalized), 'index': 0})
    try:
        yield
    finally:
        _ACTIVE_SEQUENCE.reset(token)
