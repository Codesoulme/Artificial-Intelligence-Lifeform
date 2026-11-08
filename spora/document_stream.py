"""Sample next-token examples without attention crossing document boundaries."""
import numpy as np
import torch


def trim_ignored_tail(inputs, labels):
    """Drop columns with no supervision, retaining all earlier context."""
    if inputs.ndim != 2 or inputs.shape != labels.shape:
        raise ValueError('Expected aligned two-dimensional batches')
    positions = (labels != -100).any(dim=0).nonzero()
    if not positions.numel():
        raise ValueError('No supervised targets')
    end = int(positions[-1].item()) + 1
    return inputs[:, :end].contiguous(), labels[:, :end].contiguous()


class DocumentStream:
    def __init__(self, stream, bos, eos, pad):
        self.stream = stream
        self.bos, self.eos, self.pad = int(bos), int(eos), int(pad)
        starts = np.flatnonzero(stream == self.bos)
        if len(starts) == 0 or starts[0] != 0 or int(stream[-1]) != self.eos:
            raise ValueError('Malformed document stream boundaries')
        if np.any(stream[starts[1:] - 1] != self.eos) or int(np.count_nonzero(stream == self.eos)) != len(starts):
            raise ValueError('Document boundary markers are inconsistent')
        ends = np.concatenate((starts[1:], np.array([len(stream)], dtype=starts.dtype)))
        lengths = ends - starts
        if np.any(lengths < 2):
            raise ValueError('Empty document stream segment')
        self.starts = starts
        self.ends = ends
        self.cumulative = np.cumsum(lengths - 1, dtype=np.int64)

    def batch(self, rng, batch_size, length, device):
        inputs, labels = [], []
        total = int(self.cumulative[-1])
        for _ in range(batch_size):
            sampled = rng.randrange(total)
            index = int(np.searchsorted(self.cumulative, sampled, side='right'))
            previous = int(self.cumulative[index - 1]) if index else 0
            offset = sampled - previous
            start = int(self.starts[index]) + max(0, offset - length + 1)
            end = min(int(self.ends[index]), start + length + 1)
            seq = np.asarray(self.stream[start:end], dtype=np.int64)
            if len(seq) < 2:
                raise AssertionError('Sampler returned no next-token target')
            count = len(seq) - 1
            x = np.full(length, self.pad, dtype=np.int64)
            y = np.full(length, -100, dtype=np.int64)
            x[:count], y[:count] = seq[:-1], seq[1:]
            inputs.append(x)
            labels.append(y)
        return (torch.tensor(np.stack(inputs), device=device),
                torch.tensor(np.stack(labels), device=device))
