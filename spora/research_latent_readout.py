"""Fuse separately read dialogue features into answer positions."""

import torch
from torch import nn
from torch.nn import functional as F


class LatentReadout(nn.Module):
    def __init__(self, width, rank=32):
        super().__init__()
        self.query = nn.Linear(width, rank, bias=False)
        self.key = nn.Linear(width, rank, bias=False)
        self.value = nn.Linear(width, rank, bias=False)
        self.out = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, hidden, slots):
        """Read one or more feature slots without changing earlier decoder layers."""
        if (hidden.ndim != 3 or slots.ndim != 3 or hidden.shape[0] != slots.shape[0]
                or hidden.shape[-1] != self.query.in_features or hidden.shape[-1] != slots.shape[-1]
                or slots.shape[1] == 0):
            raise ValueError('Expected aligned nonempty latent slots and hidden states')
        normalized = F.rms_norm(hidden.float(), (hidden.shape[-1],))
        slots = F.rms_norm(slots.float(), (hidden.shape[-1],))
        query, key, value = self.query(normalized), self.key(slots), self.value(slots)
        weights = (query @ key.transpose(-1, -2) * query.shape[-1] ** -.5).softmax(dim=-1)
        return hidden + self.out(weights @ value).to(hidden.dtype)
