"""Neural output conditioning for finite SPORA memory values.

The adapter is deliberately separate from decoder feature layers. It receives
the native memory logits selected by the caller and produces a vocabulary bias
for generation. It has no symbolic answer table and is inactive when no memory
logits are supplied.
"""
import math

import torch
from torch import nn
try:
    from transformers import LogitsProcessor
except ImportError:  # Core SPORA tests do not require the optional open-base stack.
    class LogitsProcessor:
        pass


class MemoryOutputAdapter(nn.Module):
    """Map native finite-memory evidence to a bounded vocabulary bias."""

    def __init__(self, value_count, vocab_size, rank=32, max_scale=8.0):
        super().__init__()
        if (type(value_count) is not int or value_count < 2 or type(vocab_size) is not int
                or vocab_size < 2 or type(rank) is not int or not 1 <= rank <= 256
                or isinstance(max_scale, bool) or not isinstance(max_scale, (int, float))
                or not math.isfinite(max_scale) or max_scale <= 0):
            raise ValueError('Invalid memory output dimensions or scale')
        self.value_count, self.vocab_size, self.rank = value_count, vocab_size, rank
        self.max_scale = float(max_scale)
        self.value_to_rank = nn.Linear(value_count, rank, bias=False)
        self.rank_to_vocab = nn.Linear(rank, vocab_size, bias=False)
        self.scale_logit = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.rank_to_vocab.weight)

    def forward(self, memory_logits):
        if (not isinstance(memory_logits, torch.Tensor) or memory_logits.ndim != 2
                or memory_logits.shape[-1] != self.value_count or not memory_logits.is_floating_point()
                or not bool(torch.isfinite(memory_logits).all())):
            raise ValueError('Finite memory logits with the configured value width required')
        probabilities = memory_logits.float().softmax(-1)
        hidden = torch.tanh(self.value_to_rank(probabilities))
        scale = self.max_scale * self.scale_logit.sigmoid()
        result = scale * self.rank_to_vocab(hidden)
        if result.shape != (memory_logits.shape[0], self.vocab_size) or not bool(torch.isfinite(result).all()):
            raise ValueError('Memory vocabulary bias must be finite and shape preserving')
        return result.to(device=memory_logits.device, dtype=memory_logits.dtype)

    def config_dict(self):
        return {'value_count': self.value_count, 'vocab_size': self.vocab_size,
                'rank': self.rank, 'max_scale': self.max_scale}


class MemoryOutputLogitsProcessor(LogitsProcessor):
    """Add one fixed memory bias to every generation step in a call."""

    def __init__(self, adapter, memory_logits, gain=1.0):
        if not isinstance(adapter, MemoryOutputAdapter):
            raise ValueError('Memory output adapter required')
        if memory_logits.ndim != 2 or memory_logits.shape[0] < 1:
            raise ValueError('Batch memory logits required')
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain <= 0:
            raise ValueError('Memory output gain must be finite and positive')
        self.bias = adapter(memory_logits).detach()
        self.gain = float(gain)
        self._applied = False

    def __call__(self, input_ids, scores):
        if (scores.ndim != 2 or scores.shape[-1] != self.bias.shape[-1]
                or (self.bias.shape[0] != 1 and scores.shape[0] not in (1, self.bias.shape[0]))):
            raise ValueError('Generation scores and memory bias batch mismatch')
        if self._applied:
            return scores
        self._applied = True
        bias = self.bias if self.bias.shape[0] == scores.shape[0] else self.bias.expand(scores.shape[0], -1)
        return scores + self.gain * bias.to(device=scores.device, dtype=scores.dtype)
