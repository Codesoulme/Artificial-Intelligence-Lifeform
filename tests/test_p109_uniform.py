import hashlib
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_p109_coverage as entry
from spora.document_stream import DocumentStream
from spora.document_stream_observation import batch_with_windows
from spora.story_coverage_run import coverage_contract, record_story, restore_coverage, sample_story
from spora.uniform_document_stream import UniformDocumentStream


class UniformTests(unittest.TestCase):
    def setUp(self):
        self.data = np.array([1, 2, 1, 3, 4, 5, 2], dtype=np.uint16)
        self.source = hashlib.sha256(self.data.tobytes()).hexdigest()
        self.docs = UniformDocumentStream(self.data, 1, 2, 0, 4)
        self.plan = {'story_sampling_mode': 'uniform', 'length': 4,
                     'respect_story_document_boundaries': True,
                     'observe_story_coverage': True, 'target_shares': {'story': 1}}

    def test_probability_and_whole_document(self):
        rng = random.Random(83)
        x, y, windows = batch_with_windows(self.docs, rng, 12000, 4, 'cpu')
        counts = [sum(w.document == i for w in windows) for i in range(2)]
        self.assertLess(abs(counts[0] - 6000), 250)
        for row, w in enumerate(windows):
            start, end = int(self.docs.starts[w.document]), int(self.docs.ends[w.document])
            seq = self.data[start:end].tolist()
            self.assertEqual((w.start, w.end), (1, len(seq)))
            self.assertEqual(x[row, :len(seq)-1].tolist(), seq[:-1])
            self.assertEqual(y[row, :len(seq)-1].tolist(), seq[1:])
            self.assertTrue((y[row, len(seq)-1:] == -100).all())

    def test_boundaries_and_selection(self):
        self.assertIsInstance(entry.make_story_documents(self.data, 1, 2, 0, self.plan), UniformDocumentStream)
        self.assertIs(type(entry.make_story_documents(self.data, 1, 2, 0, {**self.plan, 'story_sampling_mode': 'legacy'})), DocumentStream)
        for length in (0, -1, True, 3):
            with self.assertRaises(ValueError):
                UniformDocumentStream(self.data, 1, 2, 0, length)
        for data in (np.array([], dtype=np.uint16), np.array([1, 3, 1, 2]), np.array([1, 3])):
            with self.assertRaises(ValueError):
                UniformDocumentStream(data, 1, 2, 0, 4)
        rng = random.Random(1)
        state = rng.getstate()
        with self.assertRaises(ValueError):
            self.docs.batch(rng, 1, 5, 'cpu')
        self.assertEqual(state, rng.getstate())
        for bad in ({**self.plan, 'story_sampling_mode': 'other'},
                    {**self.plan, 'respect_story_document_boundaries': False}):
            with self.assertRaises(ValueError):
                entry.story_sampling_mode(bad)

    def test_observation_coverage_focus_and_resume(self):
        contract = coverage_contract(self.plan)
        rng = random.Random(52)
        other = random.Random(52)
        x, y, windows, original = sample_story(self.docs, rng, 8, 4, 'cpu', True)
        x2, y2, _, _ = sample_story(self.docs, other, 8, 4, 'cpu', False)
        self.assertTrue(torch.equal(x, x2) and torch.equal(y, y2))
        self.assertEqual(rng.getstate(), other.getstate())
        y[1, 0] = -100
        expected = {(w.document, p) for i, w in enumerate(windows) if i != 0
                    for p in range(w.start, w.end) if y[i, p-w.start] != -100}
        coverage = restore_coverage(None, contract, self.source, self.docs)
        n = record_story(coverage, self.source, windows, original, y, {0})
        actual = {(doc, pos) for (_, doc), summary in coverage.summaries().items()
                  for a, b in summary.intervals for pos in range(a, b)}
        self.assertEqual(actual, expected)
        state = rng.getstate()
        payload = {'coverage_contract': contract, 'coverage': coverage.to_dict(),
                   'counts': {'story_source_targets': n}}
        restored = restore_coverage(payload, contract, self.source, self.docs)
        self.assertEqual(restored.to_dict(), coverage.to_dict())
        resumed = random.Random()
        resumed.setstate(state)
        a = sample_story(self.docs, rng, 20, 4, 'cpu', True)
        b = sample_story(self.docs, resumed, 20, 4, 'cpu', True)
        self.assertTrue(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))
        self.assertEqual(a[2], b[2])
        self.assertEqual(rng.getstate(), resumed.getstate())

    def test_legacy_unchanged_and_identity(self):
        plan = {**self.plan, 'story_sampling_mode': 'legacy'}
        legacy = entry.make_story_documents(self.data, 1, 2, 0, plan)
        original = DocumentStream(self.data, 1, 2, 0)
        a, b = random.Random(42), random.Random(42)
        for _ in range(4):
            ax, ay, aw = batch_with_windows(legacy, a, 30, 4, 'cpu')
            bx, by, bw = batch_with_windows(original, b, 30, 4, 'cpu')
            self.assertTrue(torch.equal(ax, bx) and torch.equal(ay, by))
            self.assertEqual(aw, bw)
            self.assertEqual(a.getstate(), b.getstate())
        self.assertIn('scripts/train_p109_coverage.py', entry.SOURCES)
        self.assertIn('spora/uniform_document_stream.py', entry.SOURCES)
        self.assertNotIn('scripts/train_conversation_coverage.py', entry.SOURCES)
        self.assertEqual(entry.story_sampling_mode({}), 'legacy')

    def test_legacy_draws_follow_target_positions(self):
        _, _, windows = batch_with_windows(DocumentStream(self.data, 1, 2, 0),
                                           random.Random(83), 12000, 4, 'cpu')
        # The two documents have one and four next-token targets respectively.
        first = sum(w.document == 0 for w in windows)
        self.assertLess(abs(first - 2400), 200)


if __name__ == '__main__':
    unittest.main()
