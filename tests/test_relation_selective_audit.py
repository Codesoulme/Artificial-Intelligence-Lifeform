import unittest

import torch
from scripts.deepseek_spora_relation_selective_audit import decide, thresholds_from_calibration


class SelectiveAuditTests(unittest.TestCase):
    def test_threshold_rejects_all_seen_wrong_labels_and_keeps_high_positive(self):
        scores = torch.tensor([[.9, .03, .02, .05], [.15, .75, .04, .06],
                               [.10, .10, .7, .10], [.30, .20, .10, .40]])
        labels = ['drink', 'color', 'activity', 'reject']
        thresholds = thresholds_from_calibration(scores, labels)
        for row, label in zip(scores, labels):
            result = decide(row, thresholds)
            self.assertIn(result, ('reject', label))
        self.assertEqual(decide(scores[0], thresholds), 'drink')
        self.assertEqual(decide(torch.tensor([.29, .20, .11, .40]), thresholds), 'reject')

    def test_bad_scores_fail_closed_with_exception(self):
        with self.assertRaises(ValueError):
            decide(torch.tensor([float('nan'), 0., 0., 0.]), {})


if __name__ == '__main__':
    unittest.main()
