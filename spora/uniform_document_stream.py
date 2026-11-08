"""Uniform whole-document sampling for streams that fit a target window."""
import numpy as np

from spora.document_stream import DocumentStream


class UniformDocumentStream(DocumentStream):
    """Draw each document once per equal-probability choice.

    ``length`` counts next-token targets; a document can include one more
    token for its initial context. Long documents are rejected, not truncated.
    """

    def __init__(self, stream, bos, eos, pad, length):
        if type(length) is not int or length < 1:
            raise ValueError('Window length must be a positive integer')
        super().__init__(stream, bos, eos, pad)
        if np.any(self.ends - self.starts - 1 > length):
            raise ValueError('Document exceeds uniform sampling window')
        self.length = length
        # One draw per document. The existing sampler and observer both map
        # these unit intervals to offset zero, hence the entire document.
        self.cumulative = np.arange(1, len(self.starts) + 1, dtype=np.int64)

    def batch(self, rng, batch_size, length, device):
        if length != self.length:
            raise ValueError('Window length differs from validated length')
        return super().batch(rng, batch_size, length, device)
