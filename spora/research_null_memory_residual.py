"""Research-only trainable residual for an explicitly missing memory state."""
import torch
from torch import nn


class NullMemoryResidual(nn.Module):
    def __init__(self, width):
        super().__init__()
        if type(width) is not int or width <= 0:
            raise ValueError('Width must be positive')
        self.delta = nn.Parameter(torch.zeros(width, dtype=torch.float32))

    def forward(self, hidden):
        if hidden.ndim != 3 or hidden.shape[-1] < self.delta.numel():
            raise ValueError('Invalid hidden state')
        width = self.delta.numel()
        return torch.cat((hidden[:, :, :width] + self.delta.to(hidden.dtype), hidden[:, :, width:]), dim=-1)
