"""Research-only context-dependent residual for an explicitly empty state."""
import torch
from torch import nn
from torch.nn import functional as F


class ConditionalNullResidual(nn.Module):
    def __init__(self, width, rank=16):
        super().__init__()
        if type(width) is not int or width <= 0 or type(rank) is not int or rank <= 0:
            raise ValueError('Invalid dimensions')
        self.width = width
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden):
        if hidden.ndim != 3 or hidden.shape[-1] < self.width:
            raise ValueError('Invalid hidden state')
        source = hidden[:, :, :self.width].float()
        delta = self.up(torch.tanh(self.down(F.layer_norm(source, (self.width,)))))
        return torch.cat(((source + delta).to(hidden.dtype), hidden[:, :, self.width:]), dim=-1)
