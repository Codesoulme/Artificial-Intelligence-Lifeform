import unittest

from scripts.deepseek_spora_unknown_noise_calibration_analysis import evaluate, select_threshold


class CalibrationAnalysisTests(unittest.TestCase):
    def test_selects_midpoint_with_room_on_both_sides(self):
        samples = [{'norm': 0.1, 'known': False}, {'norm': 0.2, 'known': False},
                   {'norm': 0.8, 'known': True}, {'norm': 0.9, 'known': True}]
        threshold = select_threshold(samples)
        self.assertAlmostEqual(threshold, 0.5)
        self.assertEqual(evaluate(samples, threshold)['false_unknown'], 0)

    def test_requires_both_classes(self):
        for samples in ([], [{'norm': 1.0, 'known': True}], [{'norm': 0.0, 'known': False}]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                select_threshold(samples)


if __name__ == '__main__':
    unittest.main()
