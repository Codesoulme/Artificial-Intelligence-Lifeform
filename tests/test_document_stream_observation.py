import random
import unittest

import numpy as np
import torch

from spora.document_stream import DocumentStream
from spora.document_stream_observation import batch_with_windows, accumulate_unchanged_rows, TargetWindow
from spora.supervision_coverage import SupervisionCoverage


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

    def test_post_edit_accounting_handles_mask_and_focus_replacement(self):
        original = torch.tensor([[10, 11, 12, 13, -100], [20, 21, 22, -100, -100]])
        edited = original.clone()
        edited[0, 1] = -100
        edited[1] = torch.tensor([31, 32, 33, 34, -100])
        windows = [TargetWindow(0, 1, 5), TargetWindow(1, 3, 6)]
        coverage = SupervisionCoverage()
        self.assertEqual(accumulate_unchanged_rows(coverage, 'a' * 64, windows, original, edited, {1}), 3)
        result = coverage.summaries()[('a' * 64, 0)]
        self.assertEqual(result.intervals, ((1, 2), (3, 5)))
        self.assertEqual(result.total_positions, 3)
        self.assertNotIn(('a' * 64, 1), coverage.summaries())

    def test_bad_edits_do_not_partially_mutate_coverage(self):
        original = torch.tensor([[1, 2, -100], [3, 4, -100]])
        modified = original.clone()
        modified[1, 0] = 9
        coverage = SupervisionCoverage()
        with self.assertRaises(ValueError):
            accumulate_unchanged_rows(coverage, 'a' * 64, [TargetWindow(0, 1, 3), TargetWindow(1, 1, 3)],
                                      original, modified, set())
        self.assertEqual(coverage.summaries(), {})

    def test_sampler_and_coverage_resume_matches_continuous_sequence(self):
        data = np.array([1, 10, 11, 2, 1, 20, 21, 22, 23, 2], dtype=np.uint16)
        sampler = DocumentStream(data, 1, 2, 0)

        def advance(rng, coverage, n):
            batches = []
            for _ in range(n):
                x, y, windows = batch_with_windows(sampler, rng, 5, 3, 'cpu')
                accumulate_unchanged_rows(coverage, 'a' * 64, windows, y, y, set())
                batches.append((x.tolist(), y.tolist()))
            return batches

        full_rng, resumed_rng = random.Random(456), random.Random(456)
        full, partial = SupervisionCoverage(), SupervisionCoverage()
        expected = advance(full_rng, full, 20)
        actual = advance(resumed_rng, partial, 7)
        rng_state, coverage_state = resumed_rng.getstate(), partial.to_dict()
        restored_rng = random.Random()
        restored_rng.setstate(rng_state)
        restored = SupervisionCoverage.from_dict(coverage_state)
        actual.extend(advance(restored_rng, restored, 13))
        self.assertEqual(actual, expected)
        self.assertEqual(restored_rng.getstate(), full_rng.getstate())
        self.assertEqual(restored.summaries(), full.summaries())


if __name__ == '__main__':
    unittest.main()
