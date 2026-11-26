"""A zero-initialized cross-layer state bridge for SPORA research."""

import math

import torch
from torch import nn


class CrossLayerStateBridge(nn.Module):
    """Project a state written by an early memory site into a later layer.

    The output projection starts at zero, so attaching the bridge preserves the
    existing function until the bridge is deliberately trained.  The bridge is
    intentionally small and stateless per token; the persistent state remains
    in the SPORA memory snapshot owned by the caller.
    """

    format = 'spora-cross-layer-state-bridge-v1'

    def __init__(self, hidden_width, rank=32, gain=1.0):
        super().__init__()
        if type(hidden_width) is not int or hidden_width <= 0:
            raise ValueError('Hidden width must be a positive integer')
        if type(rank) is not int or not 1 <= rank <= hidden_width:
            raise ValueError('Bridge rank is outside the hidden width')
        if not isinstance(gain, (int, float)) or isinstance(gain, bool) or not math.isfinite(gain) or gain <= 0:
            raise ValueError('Bridge gain must be finite and positive')
        self.hidden_width = hidden_width
        self.rank = rank
        self.gain = float(gain)
        self.norm = nn.LayerNorm(hidden_width)
        self.down = nn.Linear(hidden_width, rank, bias=False)
        self.up = nn.Linear(rank, hidden_width, bias=False)
        nn.init.normal_(self.down.weight, mean=0., std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, state):
        if (not isinstance(state, torch.Tensor) or state.ndim != 3
                or state.shape[-1] != self.hidden_width or not state.is_floating_point()):
            raise ValueError('Cross-layer state must have shape [batch, sequence, hidden_width]')
        dtype = self.norm.weight.dtype
        values = state.to(device=self.norm.weight.device, dtype=dtype)
        residual = self.up(torch.tanh(self.down(self.norm(values)))) * self.gain
        return residual.to(device=state.device, dtype=state.dtype)

    def config_dict(self):
        return {'format': self.format, 'hidden_width': self.hidden_width,
                'rank': self.rank, 'gain': self.gain}

    def validate(self):
        if any(not bool(torch.isfinite(parameter.detach()).all()) for parameter in self.parameters()):
            raise ValueError('Cross-layer bridge contains a nonfinite parameter')


class FullCrossLayerStateBridge(nn.Module):
    """A full-width state reader with an identity initialization.

    This is a research control for the low-rank bridge above.  The input is
    already a memory-induced delta, so identity initialization asks whether
    the later site can use that delta directly before learning a coordinate
    change.  It is intentionally not part of the production checkpoint
    format.
    """

    format = 'spora-full-cross-layer-state-bridge-v1'

    def __init__(self, hidden_width, gain=1.0):
        super().__init__()
        if type(hidden_width) is not int or hidden_width <= 0:
            raise ValueError('Hidden width must be a positive integer')
        if not isinstance(gain, (int, float)) or isinstance(gain, bool) or not math.isfinite(gain) or gain <= 0:
            raise ValueError('Bridge gain must be finite and positive')
        self.hidden_width = hidden_width
        self.gain = float(gain)
        self.mapping = nn.Linear(hidden_width, hidden_width, bias=False)
        nn.init.eye_(self.mapping.weight)

    def forward(self, state):
        if (not isinstance(state, torch.Tensor) or state.ndim != 3
                or state.shape[-1] != self.hidden_width or not state.is_floating_point()):
            raise ValueError('Cross-layer state must have shape [batch, sequence, hidden_width]')
        values = state.to(device=self.mapping.weight.device, dtype=self.mapping.weight.dtype)
        residual = self.mapping(values) * self.gain
        return residual.to(device=state.device, dtype=state.dtype)

    def config_dict(self):
        return {'format': self.format, 'hidden_width': self.hidden_width, 'gain': self.gain,
                'initialization': 'identity'}

    def validate(self):
        if any(not bool(torch.isfinite(parameter.detach()).all()) for parameter in self.parameters()):
            raise ValueError('Cross-layer bridge contains a nonfinite parameter')
