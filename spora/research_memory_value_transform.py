"""Research-only trainable mapping after normalized memory reads."""
import torch
from torch import nn


class MemoryValueTransform(nn.Module):
    def __init__(self, width):
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError('Invalid memory value width')
        self.delta = nn.Linear(width, width, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.delta.weight)

    def forward(self, value):
        if value.shape[-1] != self.delta.in_features:
            raise ValueError('Incompatible value width')
        value = value.float()
        return value + self.delta(value)
