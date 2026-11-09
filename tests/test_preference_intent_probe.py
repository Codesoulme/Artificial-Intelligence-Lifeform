import unittest

import torch

from scripts.deepseek_preference_intent_probe import (CAL_NAMES, CAL_NEGATIVE, CAL_POSITIVE, TEST_NAMES,
    TEST_NEGATIVE, TEST_POSITIVE, TRAIN_NAMES, TRAIN_NEGATIVE, TRAIN_POSITIVE, calibrate_threshold, cases, evaluate)


class PreferenceIntentProbeTests(unittest.TestCase):
    def test_threshold_rejects_maximum_negative_including_ties(self):
        scores = torch.tensor([-1.0, 0.5, 0.5, 1.0])
        labels = torch.tensor([0, 0, 0, 1])
        threshold = calibrate_threshold(scores, labels)
        self.assertGreater(threshold, 0.5)
        rows = [{'label': int(label), 'group': 'current' if label else 'negative'} for label in labels]
        report = evaluate(scores, rows, threshold)['summary']
        self.assertEqual(report['negative']['accepted'], 0)
        self.assertEqual(report['current']['accepted'], 1)

    def test_invalid_calibration_is_rejected(self):
        with self.assertRaises(ValueError):
            calibrate_threshold(torch.tensor([1.0]), torch.tensor([1]))
        with self.assertRaises(ValueError):
            calibrate_threshold(torch.tensor([float('nan')]), torch.tensor([0]))
        with self.assertRaises(ValueError):
            evaluate(torch.tensor([1.0]), [], 0.0)

    def test_splits_do_not_share_names_or_rendered_text(self):
        groups = [(TRAIN_NAMES, TRAIN_POSITIVE, TRAIN_NEGATIVE), (CAL_NAMES, CAL_POSITIVE, CAL_NEGATIVE),
                  (TEST_NAMES, TEST_POSITIVE, TEST_NEGATIVE)]
        rendered = [{row['text'] for row in cases(*group)} for group in groups]
        for i in range(len(groups)):
            for j in range(i):
                self.assertFalse(set(groups[i][0]) & set(groups[j][0]))
                self.assertFalse(rendered[i] & rendered[j])


if __name__ == '__main__':
    unittest.main()
