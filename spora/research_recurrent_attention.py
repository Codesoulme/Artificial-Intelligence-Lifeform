"""Causal attention with two exact blocks and a recurrent feature summary."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class AttentionState:
    key: torch.Tensor
    value: torch.Tensor
    matrix: torch.Tensor
    normalizer: torch.Tensor
    length: int


class RecurrentAttention(nn.Module):
    """Use exact attention for recent blocks and a fixed-size summary for older ones."""

    def __init__(self, heads, head_dim, *, block=16, features=24):
        super().__init__()
        if min(heads, head_dim, block, features) < 1:
            raise ValueError('Attention dimensions must be positive')
        self.block = block
        self.q_map = nn.Parameter(torch.randn(heads, head_dim, features) / head_dim ** .5)
        self.k_map = nn.Parameter(self.q_map.detach().clone())
        self.gate = nn.Parameter(torch.zeros(heads, head_dim))
        self.bias = nn.Parameter(torch.full((heads,), -2.))

    def phi(self, vectors, weights):
        return F.softplus(torch.einsum('bhtd,hdf->bhtf', F.normalize(vectors, dim=-1) * vectors.shape[-1] ** .5,
                                       weights)) + 1e-5

    def forward(self, query, key, value, state=None, *, include_far=True):
        if (query.ndim != 4 or key.ndim != 4 or key.shape != value.shape or query.shape[0] != key.shape[0]
                or query.shape[-2:] != key.shape[-2:] or not query.shape[-2]
                or not key.shape[1] or query.shape[1] % key.shape[1]):
            raise ValueError('Expected aligned nonempty query, key and value')
        query = query.float()
        groups = query.shape[1] // key.shape[1]

        def expand(tensor):
            return tensor.float().repeat_interleave(groups, dim=1) if groups > 1 else tensor.float()

        if state is None:
            batch, heads, _, head_dim = query.shape
            features = self.k_map.shape[-1]
            state = AttentionState(key[:, :, :0], value[:, :, :0],
                                   query.new_zeros(batch, heads, features, head_dim),
                                   query.new_zeros(batch, heads, features), 0)
        recent_k, recent_v, matrix, normalizer, length = (
            state.key, state.value, state.matrix, state.normalizer, state.length,
        )
        outputs = []
        offset = 0
        while offset < query.shape[-2]:
            if length % self.block == 0 and recent_k.shape[-2] == 2 * self.block:
                old_features = self.phi(expand(recent_k[:, :, :self.block]), self.k_map)
                matrix = matrix + old_features.transpose(-1, -2) @ expand(recent_v[:, :, :self.block])
                normalizer = normalizer + old_features.sum(dim=-2)
                recent_k = recent_k[:, :, self.block:]
                recent_v = recent_v[:, :, self.block:]
            size = min(self.block - length % self.block, query.shape[-2] - offset)
            current = slice(offset, offset + size)
            q = query[:, :, current]
            past = recent_k.shape[-2]
            recent_k = torch.cat((recent_k, key[:, :, current]), dim=-2)
            recent_v = torch.cat((recent_v, value[:, :, current]), dim=-2)
            query_position = torch.arange(size, device=q.device) + past
            key_position = torch.arange(recent_k.shape[-2], device=q.device)
            local = F.scaled_dot_product_attention(q, expand(recent_k), expand(recent_v),
                                                  attn_mask=query_position[:, None] >= key_position[None, :])
            if include_far and length >= 2 * self.block:
                qf = self.phi(q, self.q_map)
                denom = torch.einsum('bhtf,bhf->bht', qf, normalizer).unsqueeze(-1)
                far = (qf @ matrix) / denom.clamp_min(1e-6)
                gain = torch.einsum('bhtd,hd->bht', F.normalize(q, dim=-1) * q.shape[-1] ** .5, self.gate)
                gain = (gain + self.bias[None, :, None]).sigmoid().unsqueeze(-1)
                local = torch.lerp(local, far, gain)
            outputs.append(local)
            offset += size
            length += size
        return torch.cat(outputs, dim=-2), AttentionState(recent_k, recent_v, matrix, normalizer, length)
