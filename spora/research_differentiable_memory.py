"""Research-only differentiable current-memory write and read context.

This is not a persistence or chat API.  It intentionally bypasses the runtime
snapshot's detach, so only use it with isolated state and a frozen base.
Some readers still detach memory internally; a gradient-aware adapter is needed.
"""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class DifferentiablePacketWriter(nn.Module):
    """Learn a packet from an observation and apply a differentiable delta write."""

    def __init__(self, observation_width, memory_width):
        super().__init__()
        if type(observation_width) is not int or type(memory_width) is not int or min(observation_width, memory_width) < 1:
            raise ValueError('Observation and memory widths must be positive integers')
        self.packet = nn.Linear(observation_width, memory_width, bias=False)

    def forward(self, matrix, address, observation, *, strength=1.0, active_mask=None):
        if (not isinstance(matrix, torch.Tensor) or matrix.ndim != 3
                or matrix.shape[1] != matrix.shape[2]
                or matrix.shape[1] != self.packet.out_features
                or not isinstance(address, torch.Tensor) or address.shape != (matrix.shape[0], matrix.shape[1])
                or not isinstance(observation, torch.Tensor)
                or observation.shape != (matrix.shape[0], self.packet.in_features)
                or not all(x.is_floating_point() and bool(torch.isfinite(x).all())
                           for x in (matrix, address, observation))
                or isinstance(strength, bool) or not isinstance(strength, (int, float))
                or not 0 <= strength <= 1 or bool((address.norm(dim=-1) < 1e-8).any())):
            raise ValueError('Invalid differentiable memory write')
        key = F.normalize(address.float(), dim=-1)
        target = self.packet(observation.float())
        if active_mask is not None:
            if (not isinstance(active_mask, torch.Tensor) or active_mask.dtype != torch.bool
                    or active_mask.shape != target.shape or active_mask.device != target.device):
                raise ValueError('Active mask must match the packet')
            target = target * active_mask.to(target.dtype)
        existing = torch.bmm(matrix.float(), key.unsqueeze(-1)).squeeze(-1)
        return matrix.float() + float(strength) * (target - existing).unsqueeze(-1) * key.unsqueeze(1)


@contextmanager
def differentiable_memory_context(decoder, matrices, addresses):
    """Install isolated current matrices without cloning or detaching their graphs.

    Call decoder.base directly inside the context, with cache disabled. Standard
    decoder.forward snapshots/detaches instead. Reader-specific detach operations
    remain in force and must not be mistaken for an end-to-end gradient path.
    """
    context = decoder._memory_context
    if context.get() is not None:
        raise ValueError('Nested memory calls are not supported')
    if (not isinstance(matrices, (tuple, list)) or len(matrices) != 2
            or not isinstance(addresses, (tuple, list)) or len(addresses) != 2):
        raise ValueError('Two memory matrices and addresses are required')
    width = decoder.reader.memory_width
    for matrix, address in zip(matrices, addresses):
        if (not isinstance(matrix, torch.Tensor) or matrix.shape != (1, width, width)
                or not isinstance(address, torch.Tensor) or address.shape != (1, width)
                or not matrix.is_floating_point() or not address.is_floating_point()
                or not bool(torch.isfinite(matrix).all()) or not bool(torch.isfinite(address).all())
                or bool((address.norm(dim=-1) < 1e-8).any())):
            raise ValueError('Invalid differentiable memory context')
    slots = torch.zeros(1, dtype=torch.long, device=matrices[0].device)
    token = context.set((tuple((matrix,) for matrix in matrices), tuple(addresses), None, slots))
    try:
        yield
    finally:
        context.reset(token)
