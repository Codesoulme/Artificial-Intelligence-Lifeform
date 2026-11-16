"""Research-only adaptive semantic reading during cached generation."""
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn


_SLOT = ContextVar('spora_adaptive_semantic_slot', default=None)


@contextmanager
def use_adaptive_slot(slot):
    if (not isinstance(slot, torch.Tensor) or slot.ndim not in (1, 2)
            or not slot.is_floating_point() or not bool(torch.isfinite(slot).all())):
        raise ValueError('Expected a finite semantic vector or batch')
    if _SLOT.get() is not None:
        raise ValueError('Nested adaptive slots are not supported')
    token = _SLOT.set(slot.detach())
    try:
        yield
    finally:
        _SLOT.reset(token)


class AdaptiveSemanticHead(nn.Module):
    """Learn memory and stop residuals from each decoder hidden state.

    No generated-token counter or externally supplied stop vector is used.
    The caller still decides which slot is relevant for the request.
    """

    def __init__(self, base, rank=32, gain=32.):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1 or gain <= 0:
            raise ValueError('Expected a linear head, positive rank and gain')
        self.base = base
        self.base.requires_grad_(False)
        self.gain = float(gain)
        width = base.in_features
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, rank)
        self.gates = nn.Linear(rank, 2)
        self.stop = nn.Parameter(torch.zeros(width))
        nn.init.zeros_(self.gates.weight)
        with torch.no_grad():
            self.gates.bias.copy_(torch.tensor([2., -2.]))

    def adjusted(self, hidden, slot):
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim < 2
                or hidden.shape[-1] != self.base.in_features or not isinstance(slot, torch.Tensor)
                or slot.shape[-1] != self.base.in_features):
            raise ValueError('Invalid hidden state or semantic vector')
        values = hidden.float()
        slot = slot.detach().to(device=values.device, dtype=values.dtype)
        while slot.ndim < values.ndim:
            slot = slot.unsqueeze(-2)
        if slot.ndim != values.ndim or slot.shape[0] not in (1, values.shape[0]):
            raise ValueError('Semantic vector batch does not match hidden state')
        gates = torch.sigmoid(self.gates(torch.tanh(self.down(self.norm(values)))))
        return values + self.gain * gates[..., :1] * slot + gates[..., 1:] * self.stop

    def forward(self, hidden):
        slot = _SLOT.get()
        if slot is None:
            return self.base(hidden)
        return self.base(self.adjusted(hidden, slot).to(dtype=hidden.dtype, device=hidden.device))
