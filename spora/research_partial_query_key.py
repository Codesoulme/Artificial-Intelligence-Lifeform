"""Move selected query/key coordinate pairs to a shared absolute position table."""

import torch
from torch import nn
from torch.nn import functional as F

from .pretrained_attention import SPORAFusedAttention, ValuePositionAttention


class PartialQueryKeyPositionAttention(SPORAFusedAttention):
    def __init__(self, parent, pairs, weight, bias):
        nn.Module.__init__(self)
        if not isinstance(parent, ValuePositionAttention):
            raise ValueError('Partial query/key positions require the migrated value module')
        half = parent.head_dim // 2
        if (not isinstance(pairs, (list, tuple)) or not pairs or parent.head_dim % 2
                or any(type(pair) is not int or not 0 <= pair < half for pair in pairs)
                or list(pairs) != sorted(set(pairs))):
            raise ValueError('Expected unique, sorted rotary pairs')
        coordinates = [*pairs, *(half + pair for pair in pairs)]
        reference = parent.qkv.weight
        self.register_buffer('_coordinates', torch.tensor(coordinates, device=reference.device), persistent=False)
        self.query_heads = parent.num_key_value_groups
        projection_width = len(coordinates) * (self.query_heads + 1)
        if (weight.shape != (projection_width, parent.qkv.in_features) or bias.shape != (projection_width,)
                or any(not item.is_floating_point() or not bool(torch.isfinite(item).all())
                       for item in (weight, bias))):
            raise ValueError('Invalid partial query/key projection')
        self.weight = nn.Parameter(weight.detach().to(reference), requires_grad=False)
        self.bias = nn.Parameter(bias.detach().to(reference), requires_grad=False)
        self.parent = parent
        self.pairs = tuple(pairs)
        self.group = parent.group
        self.layer_idx = parent.layer_idx
        self.head_dim = parent.head_dim
        self.scaling = parent.scaling
        self.is_causal = parent.is_causal
        self.num_key_value_groups = parent.num_key_value_groups

    @property
    def qkv(self):
        return self.parent.qkv

    @property
    def out(self):
        return self.parent.out

    def project(self, hidden_states, position_embeddings, position_ids):
        query, key, value = self.parent.project(hidden_states, position_embeddings, position_ids)
        width = self._coordinates.numel()
        position = self.parent.position(position_ids)
        projected = F.linear(hidden_states + position, self.weight, self.bias)
        query_abs, key_abs = projected.split((width * self.query_heads, width), dim=-1)
        query_abs = query_abs.unflatten(-1, (self.query_heads, width)).transpose(1, 2)
        first = self.group * self.query_heads
        last = first + self.query_heads
        new_queries = query[:, first:last].index_copy(-1, self._coordinates, query_abs)
        new_key = key[:, self.group:self.group + 1].index_copy(-1, self._coordinates, key_abs.unsqueeze(1))
        query = torch.cat((query[:, :first], new_queries, query[:, last:]), dim=1)
        key = torch.cat((key[:, :self.group], new_key, key[:, self.group + 1:]), dim=1)
        return query, key, value


def select_query_key_rows(parent, pairs, state):
    """Copy this group's query/key projection rows in half-rotation pair order."""
    if not torch.equal(parent.position.weight.detach().cpu(), state['position.weight']):
        raise ValueError('Value and query/key modules must share the same learned position table')
    width = parent.head_dim
    half = width // 2
    coordinates = [*pairs, *(pair + half for pair in pairs)]
    first = parent.group * parent.num_key_value_groups
    stop = first + parent.num_key_value_groups
    rows = [head * width + coordinate for head in range(first, stop) for coordinate in coordinates]
    start = parent.rotary.sizes[0] + parent.group * width
    rows.extend(start + coordinate for coordinate in coordinates)
    return state['qkv.weight'][rows], state['qkv.bias'][rows]
