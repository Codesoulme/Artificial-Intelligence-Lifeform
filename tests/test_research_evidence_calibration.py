import unittest
from spora.research_evidence_calibration import calibrate, confusion, rank_auc


class CalibrationTests(unittest.TestCase):
    def test_balanced_error_is_not_raw_accuracy(self):
        scores, labels = [0, 1, 2, 3, 2.5], [0, 0, 0, 0, 1]
        policies, _ = calibrate(scores, labels)
        self.assertEqual(policies['balanced'], 2.5)
        self.assertEqual(confusion(scores, labels, policies['balanced'])['fn'], 0)
        self.assertEqual(confusion(scores, labels, policies['zero_calibration_fp'])['fn'], 1)

    def test_tied_scores_and_tie_break(self):
        policies, _ = calibrate([1, 1], [0, 1])
        self.assertEqual(policies['balanced'], 1)
        self.assertEqual(rank_auc([1, 1], [0, 1]), 0.5)
        self.assertEqual(confusion([1, 1], [0, 1], policies['zero_calibration_fp'])['fp'], 0)

    def test_invalid_input_fails(self):
        for scores, labels in [([], []), ([1], [1]), ([float('nan'), 0], [1, 0]), ([0, 1], [0, 2])]:
            with self.assertRaises(ValueError):
                calibrate(scores, labels)
