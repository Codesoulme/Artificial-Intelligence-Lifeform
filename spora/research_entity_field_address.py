"""Factor a memory address into entity and field features."""

import torch
from torch import nn
from torch.nn import functional as F


class EntityFieldAddress(nn.Module):
    def __init__(self, hidden, entity_width=16, field_width=8):
        super().__init__()
        self.entity = nn.Linear(hidden, entity_width, bias=False)
        self.field = nn.Linear(hidden, field_width, bias=False)

    def forward(self, entity, context):
        name = F.normalize(self.entity(entity), dim=-1)
        field = F.normalize(self.field(context), dim=-1)
        return (name.unsqueeze(-1) * field.unsqueeze(-2)).flatten(start_dim=-2)


def revise_association(memory, address, value):
    """Replace the value read at a unit address with one residual update."""
    residual = value - address @ memory
    return memory + address.unsqueeze(-1) * residual.unsqueeze(-2)


class TokenValue:
    """Encode a bounded token sequence as signed bits, including its end marker."""

    def __init__(self, vocabulary_size, capacity=12):
        if vocabulary_size < 2 or capacity < 1:
            raise ValueError('The vocabulary and value capacity must be positive')
        self.vocabulary_size = vocabulary_size
        self.capacity = capacity
        self.bits = vocabulary_size.bit_length()
        self.powers = 1 << torch.arange(self.bits)
        self.width = self.bits * self.capacity

    def encode(self, tokens):
        if not 0 < len(tokens) < self.capacity or any(not 0 <= token < self.vocabulary_size for token in tokens):
            raise ValueError('The value must fit with room for its end marker')
        codes = torch.tensor([token + 1 for token in tokens] + [0] * (self.capacity - len(tokens)))
        return ((codes[:, None].bitwise_and(self.powers) != 0).float() * 2 - 1).flatten()

    def decode(self, vector):
        codes = ((vector.reshape(self.capacity, self.bits) > 0).long() * self.powers).sum(dim=-1).tolist()
        if 0 not in codes:
            return None
        tokens = [code - 1 for code in codes[:codes.index(0)]]
        return tokens if tokens and all(token < self.vocabulary_size for token in tokens) else None


class RecurrentFieldSlots:
    """Keep a bounded tensor of addresses and value vectors for targeted writes."""

    def __init__(self, capacity, address_width, value_width):
        if capacity < 2 or min(address_width, value_width) < 1:
            raise ValueError('Invalid slot dimensions')
        self.addresses = torch.zeros(capacity, address_width)
        self.values = torch.zeros(capacity, value_width)
        self.count = 0

    def allocate(self, key, value):
        if self.count == len(self.addresses):
            raise ValueError('No free field slot')
        self.addresses[self.count].copy_(key)
        self.values[self.count].copy_(value)
        self.count += 1

    def select(self, key, *, floor=0., gap=0., eligible=slice(None)):
        candidates = torch.arange(self.count)[eligible]
        if len(candidates) < 2:
            raise ValueError('Need at least two occupied slots for selection')
        scores, slots = (self.addresses[candidates] @ key).topk(2)
        if scores[0] < floor or scores[0] - scores[1] <= gap:
            return None
        return int(candidates[slots[0]])

    def revise(self, key, value, *, floor=0., gap=0., eligible=slice(None)):
        slot = self.select(key, floor=floor, gap=gap, eligible=eligible)
        if slot is not None:
            self.values[slot].copy_(value)
        return slot
