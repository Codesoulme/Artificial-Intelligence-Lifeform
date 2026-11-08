import random
import unittest
from collections import Counter

import numpy as np
import torch

from spora.document_chunk_stream import DocumentChunkStream
from spora.document_stream_observation import TargetWindow


BOS, EOS, PAD = 1, 2, 0


def make_stream(documents, length):
    return DocumentChunkStream(np.array([token for doc in documents for token in doc], dtype=np.int64),
                               BOS, EOS, PAD, length)


class FixedRng:
    def __init__(self, draws):
        self.draws = iter(draws)
        self.stops = []

    def randrange(self, stop):
        self.stops.append(stop)
        return next(self.draws)


class DocumentChunkStreamTests(unittest.TestCase):
    def test_partition_covers_short_long_exact_boundary_and_eos(self):
        docs = [[BOS, EOS], [BOS, 10, 11, 12, EOS],
                [BOS, *range(20, 33), EOS]]
        sampler = make_stream(docs, 4)
        actual = [sampler.window(i) for i in range(int(sampler.cumulative[-1]))]
        self.assertEqual(actual, [TargetWindow(0, 1, 2), TargetWindow(1, 1, 5),
                                  TargetWindow(2, 1, 5), TargetWindow(2, 5, 9),
                                  TargetWindow(2, 9, 13), TargetWindow(2, 13, 15)])
        seen = Counter((w.document, position) for w in actual for position in range(w.start, w.end))
        self.assertEqual(seen, Counter((i, pos) for i, doc in enumerate(docs) for pos in range(1, len(doc))))
        self.assertTrue(all(count == 1 for count in seen.values()))

    def test_each_draw_maps_to_exact_tokens_and_padding(self):
        docs = [[BOS, 10, EOS], [BOS, 20, 21, 22, 23, EOS]]
        sampler = make_stream(docs, 2)
        rng = FixedRng([0, 1, 2, 3])
        x, y, windows = sampler.batch_with_windows(rng, 4, 2, 'cpu')
        self.assertEqual(rng.stops, [4] * 4)
        self.assertEqual(windows, [TargetWindow(0, 1, 3), TargetWindow(1, 1, 3),
                                   TargetWindow(1, 3, 5), TargetWindow(1, 5, 6)])
        self.assertEqual(x.tolist(), [[BOS, 10], [BOS, 20], [21, 22], [23, PAD]])
        self.assertEqual(y.tolist(), [[10, EOS], [20, 21], [22, 23], [EOS, -100]])

    def test_sampling_uniform_and_rng_restores_for_both_interfaces(self):
        sampler = make_stream([[BOS, 10, EOS], [BOS, *range(20, 28), EOS]], 2)
        rng = random.Random(281)
        state = rng.getstate()
        x, y, windows = sampler.batch_with_windows(rng, 12000, 2, 'cpu')
        counts = Counter(windows)
        self.assertEqual(len(counts), 6)
        self.assertTrue(all(abs(count - 2000) < 200 for count in counts.values()))
        rng.setstate(state)
        replay_x, replay_y = sampler.batch(rng, 12000, 2, 'cpu')
        self.assertTrue(torch.equal(x, replay_x))
        self.assertTrue(torch.equal(y, replay_y))

    def test_length_one_and_large_document(self):
        doc = [BOS, *range(10, 10010), EOS]
        sampler = make_stream([doc], 1)
        self.assertEqual(sampler.window(9999), TargetWindow(0, 10000, 10001))
        self.assertEqual(sampler.window(10000), TargetWindow(0, 10001, 10002))
        x, y, windows = sampler.batch_with_windows(FixedRng([10000]), 1, 1, 'cpu')
        self.assertEqual((x.tolist(), y.tolist(), windows), ([[10009]], [[EOS]], [TargetWindow(0, 10001, 10002)]))

    def test_invalid_arguments(self):
        for stream in ([], [BOS], [4, EOS], [BOS, 3], [BOS, 3, BOS, EOS],
                       [BOS, EOS, EOS], [BOS, -100, EOS], [[BOS, EOS]]):
            with self.subTest(stream=stream), self.assertRaises(ValueError):
                make_stream([stream], 2)
        with self.assertRaises(ValueError):
            DocumentChunkStream(np.array([1.0, 2.0]), BOS, EOS, PAD, 2)
        for length in (0, -1, True, 1.5):
            with self.subTest(length=length), self.assertRaises(ValueError):
                make_stream([[BOS, EOS]], length)
        for markers in ((1, 1, 0), (1, 2, -100), (True, 2, 0)):
            with self.subTest(markers=markers), self.assertRaises(ValueError):
                DocumentChunkStream(np.array([BOS, EOS]), *markers, 2)
        sampler = make_stream([[BOS, EOS]], 2)
        for index in (-1, 1, True, 0.5):
            with self.subTest(index=index), self.assertRaises(ValueError):
                sampler.window(index)
        for batch_size, length in ((0, 2), (True, 2), (1, 1), (1, True)):
            with self.subTest(batch_size=batch_size, length=length), self.assertRaises(ValueError):
                sampler.batch(random.Random(), batch_size, length, 'cpu')


if __name__ == '__main__':
    unittest.main()
