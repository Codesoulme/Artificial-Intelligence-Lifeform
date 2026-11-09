"""Exact coverage of supervised positions in a document-bounded stream."""
import numpy as np
import torch


class PackedCoverage:
    def __init__(self, stream):
        self.stream = stream
        self.positions = np.zeros((len(stream.stream) + 7) // 8, dtype=np.uint8)
        self.documents = set()
        self.windows = set()
        self.presentations = 0

    def add(self, spans):
        for document, start, end in spans:
            if (not 0 <= document < len(self.stream.starts)
                    or not int(self.stream.starts[document]) <= start < end < int(self.stream.ends[document])):
                raise ValueError('Invalid supervised span')
        for document, start, end in spans:
            # A window's input at start predicts positions start+1 through end.
            positions = np.arange(start + 1, end + 1, dtype=np.int64)
            np.bitwise_or.at(self.positions, positions // 8, (1 << (positions % 8)).astype(np.uint8))
            self.documents.add(document)
            self.windows.add((document, start, end))
            self.presentations += end - start

    def metrics(self):
        return {'target_presentations': self.presentations,
                'distinct_target_positions': int(np.unpackbits(self.positions).sum()),
                'distinct_documents': len(self.documents), 'distinct_windows': len(self.windows)}

    def snapshot(self):
        return {'positions': torch.from_numpy(self.positions.copy()),
                'windows': [list(window) for window in sorted(self.windows)],
                'presentations': self.presentations}

    @classmethod
    def restore(cls, stream, payload):
        if not isinstance(payload, dict) or set(payload) != {'positions', 'windows', 'presentations'}:
            raise ValueError('Invalid coverage state')
        result = cls(stream)
        positions, windows, presentations = payload['positions'], payload['windows'], payload['presentations']
        if (not isinstance(positions, torch.Tensor) or positions.dtype != torch.uint8
                or positions.ndim != 1 or positions.numel() != len(result.positions)
                or not isinstance(windows, list) or type(presentations) is not int or presentations < 0):
            raise ValueError('Invalid coverage state')
        if any(not isinstance(w, (tuple, list)) or len(w) != 3
               or any(type(v) is not int for v in w) for w in windows):
            raise ValueError('Invalid coverage windows')
        result.add(windows)
        if (not np.array_equal(result.positions, positions.cpu().numpy())
                or len(result.windows) != len(windows) or result.presentations > presentations):
            raise ValueError('Inconsistent coverage state')
        result.presentations = presentations
        return result
