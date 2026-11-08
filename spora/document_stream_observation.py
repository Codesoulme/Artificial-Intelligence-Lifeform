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


def accumulate_unchanged_rows(coverage, source, windows, sampled_labels, final_labels, replaced_rows):
    """Account only unchanged, still-supervised targets after batch edits.

    Replaced rows must be explicitly supplied, including when replacement tokens
    happen to equal the original. Their new source needs separate accounting.
    Any other target value change is rejected. Call only once per committed batch.
    """
    if sampled_labels.ndim != 2 or sampled_labels.shape != final_labels.shape:
        raise ValueError('Expected aligned label batches')
    if len(windows) != sampled_labels.shape[0]:
        raise ValueError('Missing sampling windows')
    if not isinstance(replaced_rows, (set, frozenset)) or any(
        type(row) is not int or not 0 <= row < len(windows) for row in replaced_rows
    ):
        raise ValueError('Invalid replacement row set')
    pending = []
    for row, window in enumerate(windows):
        if row in replaced_rows:
            continue
        before, after = sampled_labels[row], final_labels[row]
        active = after != -100
        if bool((active & ((before == -100) | (before != after))).any()):
            raise ValueError('Unattributed label modification')
        count = window.end - window.start
        if count < 1 or count > len(before) or bool((before[:count] == -100).any()) or bool((before[count:] != -100).any()):
            raise ValueError('Sampled labels do not match window')
        positions = active[:count].nonzero(as_tuple=False).flatten().tolist()
        if not positions:
            continue
        start = previous = positions[0]
        for position in positions[1:]:
            if position != previous + 1:
                pending.append((window.document, window.start + start, window.start + previous + 1))
                start = position
            previous = position
        pending.append((window.document, window.start + start, window.start + previous + 1))
    # Validate all records before changing the supplied accumulator.
    from spora.supervision_coverage import SupervisionCoverage
    validated = SupervisionCoverage()
    for document, start, end in pending:
        validated.add(source, document, start, end)
    for document, start, end in pending:
        coverage.add(source, document, start, end)
    return sum(end - start for _, start, end in pending)
