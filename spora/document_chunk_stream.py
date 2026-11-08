"""Uniform sampling of non-overlapping next-token target chunks.

Target coordinates are document-relative, half-open token offsets, including
EOS. Each chunk starts with exactly the preceding token as its initial context;
later chunks therefore have less history than the old overlapping sampler.
This sampler is not distributionally equivalent to that sampler.
"""
import numpy as np
import torch

from spora.document_stream import DocumentStream
from spora.document_stream_observation import TargetWindow


class DocumentChunkStream(DocumentStream):
    """Partition each document's targets once, then sample chunks uniformly.

    ``length`` is the maximum number of targets per chunk. The caller owns
    the RNG, so its usual getstate/setstate restores the sequence of draws.
    The underlying stream must remain unchanged for the lifetime of this object.
    """

    def __init__(self, stream, bos, eos, pad, length):
        if type(length) is not int or length < 1:
            raise ValueError('Window length must be a positive integer')
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in (bos, eos, pad)):
            raise ValueError('Boundary and padding tokens must be integers')
        if bos == eos or -100 in (bos, eos, pad):
            raise ValueError('Invalid boundary or padding tokens')
        stream = np.asarray(stream)
        if stream.ndim != 1 or stream.size == 0 or stream.dtype.kind not in 'iu':
            raise ValueError('Expected a nonempty one-dimensional integer stream')
        if np.any(stream == -100) or (stream.dtype.kind == 'u' and np.any(stream > np.iinfo(np.int64).max)):
            raise ValueError('Stream contains an invalid token')
        super().__init__(stream, bos, eos, pad)
        self.length = length
        targets = self.ends - self.starts - 1
        counts = np.ones_like(targets) if length >= int(targets.max()) else (targets - 1) // length + 1
        self.cumulative = np.cumsum(counts, dtype=np.int64)

    def window(self, index):
        """Return the target coordinates of a chunk in stream order."""
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < int(self.cumulative[-1]):
            raise ValueError('Chunk index out of range')
        document = int(np.searchsorted(self.cumulative, index, side='right'))
        previous = int(self.cumulative[document - 1]) if document else 0
        start = 1 + (int(index) - previous) * self.length
        end = min(start + self.length, int(self.ends[document] - self.starts[document]))
        return TargetWindow(document, start, end)

    def batch_with_windows(self, rng, batch_size, length, device):
        """Draw one chunk per row and return inputs, labels and coordinates."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Batch size must be a positive integer')
        if type(length) is not int or length != self.length:
            raise ValueError('Window length differs from validated length')
        inputs = np.full((batch_size, length), self.pad, dtype=np.int64)
        labels = np.full((batch_size, length), -100, dtype=np.int64)
        windows = []
        for row in range(batch_size):
            window = self.window(rng.randrange(int(self.cumulative[-1])))
            base = int(self.starts[window.document])
            sequence = self.stream[base + window.start - 1:base + window.end]
            count = window.end - window.start
            inputs[row, :count] = sequence[:-1]
            labels[row, :count] = sequence[1:]
            windows.append(window)
        return (torch.as_tensor(inputs, device=device),
                torch.as_tensor(labels, device=device), windows)

    def batch(self, rng, batch_size, length, device):
        inputs, labels, _ = self.batch_with_windows(rng, batch_size, length, device)
        return inputs, labels
