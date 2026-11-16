"""Research-only smooth tail for a frozen learned position table.

This is neither ALiBi nor RoPE interpolation. Its materialized tail can be
copied to an ordinary position table for evaluation and persistence.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class SmoothPositionTail(nn.Module):
    def __init__(self, prefix, total_length):
        super().__init__()
        if (not isinstance(prefix, torch.Tensor) or prefix.ndim != 2 or not prefix.is_floating_point()
                or not prefix.shape[0] or not prefix.shape[1] or not bool(torch.isfinite(prefix).all())
                or type(total_length) is not int or total_length <= prefix.shape[0]):
            raise ValueError('Finite position prefix and larger total length required')
        self.register_buffer('prefix', prefix.detach().clone())
        t = torch.linspace(0., 1., total_length - prefix.shape[0], device=prefix.device, dtype=prefix.dtype)
        basis = torch.stack((torch.ones_like(t), t, torch.sin(math.pi * t),
                             torch.sin(2 * math.pi * t), torch.sin(4 * math.pi * t)), dim=-1)
        self.register_buffer('basis', basis)
        self.coefficients = nn.Parameter(torch.zeros((basis.shape[1], prefix.shape[1]),
                                                     device=prefix.device, dtype=prefix.dtype))

    @property
    def weight(self):
        return torch.cat((self.prefix, self.basis @ self.coefficients), dim=0)

    def forward(self, positions):
        return F.embedding(positions, self.weight)
