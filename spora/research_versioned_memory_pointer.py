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


class VersionCoverageBias:
    """Favor continuing a partially emitted memory value during greedy decoding."""

    def __init__(self, value, strength=8.):
        self.value = tuple(value)
        self.strength = strength
        self.prompt_length = None
        self.applied = 0

    def next_token(self, generated):
        length = len(self.value)
        if length < 2 or not generated:
            return None
        if any(generated[start:start + length] == self.value for start in range(len(generated) - length + 1)):
            return None
        for covered in range(min(length - 1, len(generated)), 0, -1):
            if generated[-covered:] == self.value[:covered]:
                return self.value[covered]
        return None

    def __call__(self, input_ids, scores):
        if len(self.value) < 2:
            return scores
        if input_ids.shape[0] != 1 or scores.shape[0] != 1:
            raise ValueError('Version coverage supports one greedy generation at a time')
        if self.prompt_length is None:
            self.prompt_length = input_ids.shape[1]
        generated = tuple(input_ids[0, self.prompt_length:].tolist())
        token = self.next_token(generated)
        if token is None:
            return scores
        biased = scores.clone()
        biased[0, token] += self.strength
        self.applied += 1
        return biased


class VersionReplyActions(nn.Module):
    """Learn when to emit one complete stored value during ordinary generation."""

    def __init__(self, hidden_size, scaffold_tokens):
        super().__init__()
        self.register_buffer('scaffold_tokens', torch.as_tensor(scaffold_tokens, dtype=torch.long))
        self.project = nn.Linear(hidden_size, 96)
        self.progress = nn.Embedding(2, 96)
        self.actions = nn.Linear(96, 1 + len(scaffold_tokens))

    def forward(self, hidden, copied):
        hidden = F.normalize(hidden.float(), dim=-1) * hidden.shape[-1] ** .5
        scores = self.actions(F.gelu(self.project(hidden) + self.progress(copied.long())))
        return torch.cat((scores[..., :1].masked_fill(copied[..., None], -torch.inf), scores[..., 1:]), dim=-1)

    def choose(self, hidden, native_logits, *, copied, room_for_copy, memory_fraction):
        progress = torch.as_tensor(copied, device=hidden.device)
        actions = self(hidden, progress)
        if not room_for_copy:
            actions[0] = -torch.inf
        actions = actions.softmax(-1) * memory_fraction
        native = native_logits.float().softmax(-1) * (1 - memory_fraction)
        native.scatter_add_(0, self.scaffold_tokens, actions[1:])
        probability, token = native.max(0)
        return None if bool(actions[0] > probability) else int(token)
