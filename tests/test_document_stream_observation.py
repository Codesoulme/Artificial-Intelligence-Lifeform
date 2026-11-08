import random
import unittest

import numpy as np
import torch

from spora.document_stream import DocumentStream
from spora.document_stream_observation import batch_with_windows


class ObservationTests(unittest.TestCase):
    def test_observed_batch_and_rng_match_original(self):
        data = np.array([1, 10, 11, 2, 1, 20, 21, 22, 23, 2], dtype=np.uint16)
        sampler = DocumentStream(data, bos=1, eos=2, pad=0)
        for length in (1, 3, 8):
            original_rng, observed_rng = random.Random(678), random.Random(678)
            x, y = sampler.batch(original_rng, 50, length, 'cpu')
            ox, oy, windows = batch_with_windows(sampler, observed_rng, 50, length, 'cpu')
            torch.testing.assert_close(x, ox)
            torch.testing.assert_close(y, oy)
            self.assertEqual(original_rng.getstate(), observed_rng.getstate())
            for row, window in enumerate(windows):
                first = int(sampler.starts[window.document])
                expected = data[first + window.start:first + window.end].astype(np.int64).tolist()
                self.assertEqual(y[row][y[row] != -100].tolist(), expected)

    def test_changed_sampler_protocol_fails_closed(self):
        class ChangedSampler:
            cumulative = np.array([1])

            def batch(self, rng, batch_size, length, device):
                return torch.zeros((1, 1)), torch.zeros((1, 1))

        with self.assertRaises(RuntimeError):
            batch_with_windows(ChangedSampler(), random.Random(1), 1, 1, 'cpu')

    def test_changed_labels_fail_closed(self):
        class ChangedStream(DocumentStream):
            def batch(self, *args):
                x, y = super().batch(*args)
                y[0, 0] += 1
                return x, y

        sampler = ChangedStream(np.array([1, 10, 11, 2], dtype=np.uint16), 1, 2, 0)
        with self.assertRaises(RuntimeError):
            batch_with_windows(sampler, random.Random(7), 1, 3, 'cpu')


if __name__ == '__main__':
    unittest.main()
