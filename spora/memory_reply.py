"""Checkpointable output conditioning for complete known memory values."""

import math

import torch
from torch import nn

from .memory_output import LogitsProcessor, MemoryOutputAdapter


class MemoryReplyAdapter(nn.Module):
    format = 'spora-memory-reply-v1'

    def __init__(self, value_count, vocab_size, steps, rank=8, max_scale=8.):
        super().__init__()
        if type(steps) is not int or not 1 <= steps <= 16:
            raise ValueError('Expected a bounded number of memory reply steps')
        self.heads = nn.ModuleList(
            MemoryOutputAdapter(value_count, vocab_size, rank=rank, max_scale=max_scale) for _ in range(steps)
        )
        self.value_count = value_count

    def forward(self, memory_logits):
        return torch.stack([head(memory_logits) for head in self.heads], dim=1)

    def config_dict(self):
        return {**self.heads[0].config_dict(), 'format': self.format, 'steps': len(self.heads)}


class MemoryReplyLogitsProcessor(LogitsProcessor):
    """Apply each learned output bias at its relative generation position."""

    def __init__(self, adapter, evidence, gain):
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain <= 0:
            raise ValueError('Memory output gain must be finite and positive')
        if not isinstance(evidence, torch.Tensor) or evidence.ndim != 2 or evidence.shape[0] < 1:
            raise ValueError('Batch memory logits required')
        with torch.no_grad():
            self.bias = adapter(evidence).detach()
        self.prefix_length = None
        self.gain = float(gain)

    def __call__(self, input_ids, scores):
        batch, steps, width = self.bias.shape
        if (input_ids.ndim != 2 or scores.ndim != 2 or scores.shape[0] != input_ids.shape[0]
                or scores.shape[1] != width or scores.shape[0] == 0 or scores.shape[0] % batch != 0):
            raise ValueError('Generation scores and memory replies do not align')
        if self.prefix_length is None:
            self.prefix_length = input_ids.shape[1]
        position = input_ids.shape[1] - self.prefix_length
        if position < 0:
            raise ValueError('Generation sequence is shorter than the memory prefix')
        if position >= steps:
            return scores
        bias = self.bias[:, position].to(scores).repeat_interleave(scores.shape[0] // batch, dim=0)
        return scores + self.gain * bias
