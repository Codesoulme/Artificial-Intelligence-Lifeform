"""Experimental token-causal gate on an existing neural memory reader.

This module is research-only. It does not select or authenticate an entity,
decide whether a memory exists, or establish access authorization.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class SelectiveMemoryReader(nn.Module):
    """Gate the reader residual independently at each already-seen token."""

    def __init__(self, reader, hidden_width, threshold=0.5):
        super().__init__()
        if (not isinstance(reader, nn.Module) or type(hidden_width) is not int or hidden_width < 1
                or isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError('Invalid selective reader configuration')
        self.reader = reader
        self.gates = nn.ModuleList([nn.Linear(hidden_width, 1) for _ in range(2)])
        self.hidden_width = hidden_width
        self.threshold = float(threshold)
        self.enabled = True

    def forward(self, hidden, matrices, site, address=None):
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
                or hidden.shape[-1] != self.hidden_width or type(site) is not int or site not in (0, 1)):
            raise ValueError('Invalid selective memory input')
        if not self.enabled:
            return self.reader(hidden, matrices, site, address)
        features = F.layer_norm(hidden.float(), (self.hidden_width,))
        gate = self.gates[site]
        scores = gate(features.to(device=gate.weight.device)).sigmoid()
        allow = (scores >= self.threshold).to(device=hidden.device)
        injected = self.reader(hidden, matrices, site, address)
        return torch.where(allow, injected, hidden)
