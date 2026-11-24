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
    history_key: torch.Tensor | None = None

    def tensors(self):
        base = (self.key, self.value, self.matrix, self.normalizer)
        return base if self.history_key is None else (*base, self.history_key)


class RecurrentAttention(nn.Module):
    """Use exact attention for recent blocks and a fixed-size summary for older ones."""

    update_rule = 'sum'

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

    def mix_gain(self, query):
        unit = F.normalize(query, dim=-1) * query.shape[-1] ** .5
        gain = torch.einsum('bhtd,hd->bht', unit, self.gate)
        return (gain + self.bias[None, :, None]).sigmoid().unsqueeze(-1)

    def fold(self, keys, values, matrix, normalizer):
        return matrix + keys.transpose(-1, -2) @ values, normalizer + keys.sum(dim=-2)

    def prefixes(self, keys, values):
        return (keys.transpose(-1, -2) @ values).cumsum(dim=-3), keys.sum(dim=-2).cumsum(dim=-2)

    def normalize_read(self, weights, matrix_read, normalizer, *, blocked):
        denom = (weights @ normalizer.unsqueeze(-1) if blocked else
                 torch.einsum('bhtf,bhf->bht', weights, normalizer).unsqueeze(-1))
        return matrix_read / denom.clamp_min(1e-6)

    def prefill(self, query, key, value, groups, *, include_far, history):
        size, block = query.shape[-2], self.block
        blocks = (size + block - 1) // block

        def block_view(tensor):
            return F.pad(tensor, (0, 0, 0, blocks * block - size)).unflatten(-2, (blocks, block))

        def previous_and_current(tensor):
            previous = F.pad(tensor[:, :, :-1], (0, 0, 0, 0, 1, 0))
            return torch.cat((previous, tensor), dim=-2)

        q = block_view(query)
        k = block_view(key.float().repeat_interleave(groups, dim=1))
        v = block_view(value.float().repeat_interleave(groups, dim=1))
        row = torch.arange(block, device=query.device)
        column = torch.arange(2 * block, device=query.device)
        mask = (row[:, None] + block >= column[None, :]).expand(blocks, -1, -1).clone()
        mask[0, :, :block] = False
        local = F.scaled_dot_product_attention(q, previous_and_current(k), previous_and_current(v), attn_mask=mask)
        history_query, history_key = history if history is not None else (query, key)
        summary_keys = k if history is None else block_view(history_key.float().repeat_interleave(groups, dim=1))
        old_k, old_v = summary_keys[:, :, :-2], v[:, :, :-2]
        phi = self.phi(old_k.flatten(2, 3), self.k_map).unflatten(-2, (blocks - 2, block))
        matrices, normalizers = self.prefixes(phi, old_v)
        if include_far:
            recent_queries = q[:, :, 2:].flatten(2, 3)
            summary_queries = recent_queries if history is None else block_view(history_query)[:, :, 2:].flatten(2, 3)
            qf = self.phi(summary_queries, self.q_map).unflatten(-2, (blocks - 2, block))
            far = self.normalize_read(qf, qf @ matrices, normalizers, blocked=True)
            gain = self.mix_gain(recent_queries).unflatten(-2, (blocks - 2, block))
            local = torch.cat((local[:, :, :2], torch.lerp(local[:, :, 2:], far, gain)), dim=2)
        start = (blocks - 2) * block
        state = AttentionState(key[:, :, start:].clone(), value[:, :, start:].clone(),
                               matrices[:, :, -1].clone(), normalizers[:, :, -1].clone(), size,
                               None if history is None else history_key[:, :, start:].clone())
        return local.flatten(2, 3)[:, :, :size], state

    def forward(self, query, key, value, state=None, *, include_far=True, batched_prefill=False, history=None):
        if (query.ndim != 4 or key.ndim != 4 or key.shape != value.shape or query.shape[0] != key.shape[0]
                or query.shape[-2:] != key.shape[-2:] or not query.shape[-2]
                or not key.shape[1] or query.shape[1] % key.shape[1]):
            raise ValueError('Expected aligned nonempty query, key and value')
        query = query.float()
        if history is not None:
            h_query, h_key = history
            if h_query.shape != query.shape or h_key.shape != key.shape:
                raise ValueError('History projections must match the local query and key shapes')
            history = h_query.float(), h_key
        if state is not None and (state.history_key is None) != (history is None):
            raise ValueError('History projection changed during the stream')
        history_query, history_key = history if history is not None else (query, key)
        groups = query.shape[1] // key.shape[1]
        if batched_prefill and state is None and 4 * self.block <= query.shape[-2] <= 2048:
            return self.prefill(query, key, value, groups, include_far=include_far, history=history)

        def expand(tensor):
            return tensor.float().repeat_interleave(groups, dim=1) if groups > 1 else tensor.float()

        if state is None:
            batch, heads, _, head_dim = query.shape
            features = self.k_map.shape[-1]
            state = AttentionState(key[:, :, :0], value[:, :, :0],
                                   query.new_zeros(batch, heads, features, head_dim),
                                   query.new_zeros(batch, heads, features), 0,
                                   None if history is None else history_key[:, :, :0])
        recent_k, recent_v, matrix, normalizer, length = (
            state.key, state.value, state.matrix, state.normalizer, state.length,
        )
        recent_history = state.history_key
        outputs = []
        offset = 0
        while offset < query.shape[-2]:
            if length % self.block == 0 and recent_k.shape[-2] == 2 * self.block:
                folded = recent_k if recent_history is None else recent_history
                old_features = self.phi(expand(folded[:, :, :self.block]), self.k_map)
                matrix, normalizer = self.fold(old_features, expand(recent_v[:, :, :self.block]), matrix, normalizer)
                recent_k = recent_k[:, :, self.block:]
                recent_v = recent_v[:, :, self.block:]
                if recent_history is not None:
                    recent_history = recent_history[:, :, self.block:]
            size = min(self.block - length % self.block, query.shape[-2] - offset)
            current = slice(offset, offset + size)
            q = query[:, :, current]
            past = recent_k.shape[-2]
            recent_k = torch.cat((recent_k, key[:, :, current]), dim=-2)
            recent_v = torch.cat((recent_v, value[:, :, current]), dim=-2)
            if recent_history is not None:
                recent_history = torch.cat((recent_history, history_key[:, :, current]), dim=-2)
            query_position = torch.arange(size, device=q.device) + past
            key_position = torch.arange(recent_k.shape[-2], device=q.device)
            local = F.scaled_dot_product_attention(q, expand(recent_k), expand(recent_v),
                                                  attn_mask=query_position[:, None] >= key_position[None, :])
            if include_far and length >= 2 * self.block:
                qf = self.phi(history_query[:, :, current], self.q_map)
                far = self.normalize_read(qf, qf @ matrix, normalizer, blocked=False)
                local = torch.lerp(local, far, self.mix_gain(q))
            outputs.append(local)
            offset += size
            length += size
        return torch.cat(outputs, dim=-2), AttentionState(recent_k, recent_v, matrix, normalizer, length, recent_history)


class DeltaRecurrentAttention(RecurrentAttention):
    """Fit values with a bounded block delta update on normalized feature keys."""

    update_rule = 'block_delta'

    def phi(self, vectors, weights):
        features = super().phi(vectors, weights)
        return F.normalize(features - features.mean(dim=-1, keepdim=True), dim=-1, eps=1e-6)

    def fold(self, keys, values, matrix, normalizer):
        residual = values - keys @ matrix
        # Unit-length rows bound the spectral norm of K.T @ K by the block size.
        update = (keys.transpose(-1, -2) @ residual) / keys.shape[-2]
        return matrix + update, normalizer

    def prefixes(self, keys, values):
        batch, heads, _, _, features = keys.shape
        matrix = keys.new_zeros(batch, heads, features, values.shape[-1])
        normalizer = keys.new_zeros(batch, heads, features)
        states = []
        for block_keys, block_values in zip(keys.unbind(dim=2), values.unbind(dim=2), strict=True):
            matrix, _ = self.fold(block_keys, block_values, matrix, normalizer)
            states.append(matrix)
        return torch.stack(states, dim=2), normalizer.unsqueeze(2).expand(-1, -1, len(states), -1)

    def normalize_read(self, weights, matrix_read, normalizer, *, blocked):
        return matrix_read


class GatedDeltaRecurrentAttention(DeltaRecurrentAttention):
    """Learn separate retention and residual-write strengths for each history block."""

    update_rule = 'gated_block_delta'

    def __init__(self, heads, head_dim, *, block=16, features=24):
        super().__init__(heads, head_dim, block=block, features=features)
        self.history_gates = nn.Parameter(torch.zeros(heads, features, 2))
        self.history_bias = nn.Parameter(torch.tensor([-5., 5.]).expand(heads, 2).clone())

    @staticmethod
    def gate_input(keys):
        return keys.mean(dim=-2) * keys.shape[-1] ** .5

    def gate_logits(self, gate_input):
        return torch.einsum('bhf,hfg->bhg', gate_input, self.history_gates) + self.history_bias

    def fold(self, keys, values, matrix, normalizer):
        gates = self.gate_logits(self.gate_input(keys))
        forget, write = gates.sigmoid().unbind(dim=-1)
        retained = matrix * (1 - forget[..., None, None])
        residual = values - keys @ retained
        update = (keys.transpose(-1, -2) @ residual) / keys.shape[-2]
        return retained + write[..., None, None] * update, normalizer


class SplitGatedDeltaRecurrentAttention(GatedDeltaRecurrentAttention):
    """Use an independent key-directed erase strength at each block update."""

    update_rule = 'split_gated_block_delta'

    def __init__(self, heads, head_dim, *, block=16, features=24):
        super().__init__(heads, head_dim, block=block, features=features)
        self.erase_offset = nn.Parameter(torch.zeros(heads, features))
        self.erase_bias = nn.Parameter(torch.zeros(heads))

    def fold(self, keys, values, matrix, normalizer):
        gate_input = self.gate_input(keys)
        forget_logit, write_logit = self.gate_logits(gate_input).unbind(dim=-1)
        erase_logit = write_logit + torch.einsum('bhf,hf->bh', gate_input, self.erase_offset) + self.erase_bias
        retained = matrix * (1 - forget_logit.sigmoid()[..., None, None])
        write = write_logit.sigmoid()[..., None, None]
        erase = erase_logit.sigmoid()[..., None, None]
        # Match the tied update at initialization and let the erase gate learn
        # to remove a different amount along the normalized block keys.
        residual = write * values - erase * (keys @ retained)
        update = (keys.transpose(-1, -2) @ residual) / keys.shape[-2]
        return retained + update, normalizer


RECURRENT_UPDATE_RULES = {core.update_rule: core for core in (
    RecurrentAttention, DeltaRecurrentAttention, GatedDeltaRecurrentAttention, SplitGatedDeltaRecurrentAttention,
)}


def is_append_only_causal_mask(mask, cache_position, query_length):
    """Recognize the standard causal SDPA mask for an unpadded cache append."""
    if (cache_position is None or cache_position.shape != (query_length,)
            or mask.ndim != 4 or mask.shape[:3] != (1, 1, query_length)
            or mask.shape[-1] < query_length):
        return False
    width = mask.shape[-1]
    positions = torch.arange(width, device=mask.device)
    if not torch.equal(cache_position, positions[-query_length:]):
        return False
    allowed = positions[None, :] <= cache_position[:, None]
    actual = mask[0, 0]
    if actual.dtype == torch.bool:
        return torch.equal(actual, allowed)
    if not actual.is_floating_point():
        return False
    blocked = actual.isneginf() | actual.eq(torch.finfo(actual.dtype).min)
    return bool(torch.where(allowed, actual.eq(0), blocked).all())
