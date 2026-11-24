"""A research decoder head that mixes native generation with version-local copying."""

import torch
from torch import nn
from torch.nn import functional as F


class VersionMemoryPointer(nn.Module):
    """Predict which memory slot or ordinary continuation to favor at each token."""

    def __init__(self, hidden_size, scaffold_tokens, slots=4):
        super().__init__()
        self.slots = slots
        self.register_buffer('scaffold_tokens', torch.as_tensor(scaffold_tokens, dtype=torch.long))
        self.actions = nn.Sequential(nn.Linear(hidden_size, 96), nn.GELU(),
                                     nn.Linear(96, slots + len(scaffold_tokens)))

    def forward(self, hidden, lengths):
        scores = self.actions(F.normalize(hidden.float(), dim=-1) * hidden.shape[-1] ** .5)
        available = torch.arange(self.slots, device=scores.device) < lengths[..., None]
        return torch.cat((scores[..., :self.slots].masked_fill(~available, -torch.inf),
                          scores[..., self.slots:]), dim=-1)

    def mix(self, hidden, native_logits, memory_tokens, *, memory_fraction):
        """Mix distributions over the full vocabulary without masking native tokens."""
        length = len(memory_tokens)
        if not length:
            return native_logits
        scores = self(hidden, hidden.new_full(hidden.shape[:-1], length))
        weights = torch.cat((scores[..., :length], scores[..., self.slots:]), dim=-1).softmax(-1)
        positions = torch.cat((memory_tokens, self.scaffold_tokens)).expand_as(weights)
        native = native_logits.float().softmax(-1) * (1 - memory_fraction)
        return native.scatter_add(-1, positions, weights * memory_fraction).log()
