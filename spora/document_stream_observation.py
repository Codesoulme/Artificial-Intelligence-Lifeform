"""Optional sampling coordinates for a document stream, without extra RNG draws.

Coordinates describe sampled targets before any later replacement or masking.
Callers must discard or adjust them after such operations before accounting.
This adapter is not connected to the training loop.
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TargetWindow:
    document: int
    start: int
    end: int


class _DrawObserver:
    def __init__(self, rng):
        self.rng = rng
        self.draws = []

    def randrange(self, stop):
        value = self.rng.randrange(stop)
        self.draws.append((stop, value))
        return value


def batch_with_windows(sampler, rng, batch_size, length, device):
    """Return the original batch and per-row target offsets within documents."""
    observer = _DrawObserver(rng)
    x, y = sampler.batch(observer, batch_size, length, device)
    total = int(sampler.cumulative[-1])
    if len(observer.draws) != batch_size or any(stop != total for stop, _ in observer.draws):
        raise RuntimeError('Unexpected sampling protocol; coordinates unavailable')
    windows = []
    for row, (_, sampled) in enumerate(observer.draws):
        index = int(np.searchsorted(sampler.cumulative, sampled, side='right'))
        previous = int(sampler.cumulative[index - 1]) if index else 0
        offset = sampled - previous
        start = max(0, offset - length + 1)
        document_length = int(sampler.ends[index] - sampler.starts[index])
        end = min(document_length, start + length + 1)
        target = TargetWindow(index, start + 1, end)
        count = target.end - target.start
        if not bool((y[row, :count] != -100).all()) or not bool((y[row, count:] == -100).all()):
            raise RuntimeError('Supervision does not match sampled coordinates')
        base = int(sampler.starts[index])
        tokens = torch.as_tensor(np.asarray(sampler.stream[base + start:base + end], dtype=np.int64), device=y.device)
        if not torch.equal(x[row, :count], tokens[:-1]) or not torch.equal(y[row, :count], tokens[1:]):
            raise RuntimeError('Sampled tokens do not match coordinates')
        windows.append(target)
    return x, y, windows
