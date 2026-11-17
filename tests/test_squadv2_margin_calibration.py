import unittest
from scripts.deepseek_squadv2_margin_calibration_20261117 import choose_threshold, measure


class MarginCalibrationTests(unittest.TestCase):
    def test_constraint_and_exact_tie(self):
        rows = [{'kind': 'known', 'margin': 2., 'best_exact': True},
                {'kind': 'known', 'margin': 1., 'best_exact': False},
                {'kind': 'unknown', 'margin': 1.5, 'best_exact': None}]
        threshold = choose_threshold(rows, 0)
        self.assertEqual(threshold, 1.5)
        self.assertEqual(measure(rows, threshold)['known']['exact'], 1)
        self.assertEqual(measure(rows, threshold)['unknown']['selected'], 0)

    def test_empty_class_rejected(self):
        with self.assertRaises(ValueError):
            choose_threshold([{'kind': 'known', 'margin': 1., 'best_exact': True}], 0)
