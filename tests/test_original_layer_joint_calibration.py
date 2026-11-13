import unittest

from scripts.deepseek_spora_original_layer_joint_calibration import joint_single_threshold


class JointCalibrationTests(unittest.TestCase):
    def test_considers_only_non_single_rows_passing_read_gate(self):
        rows = [
            {'split': 'calibration', 'single': 0, 'single_score': 9.0, 'read_score': -4.0},
            {'split': 'calibration', 'single': 0, 'single_score': 0.5, 'read_score': 1.0},
            {'split': 'calibration', 'single': 1, 'single_score': 8.0, 'read_score': 1.0},
            {'split': 'new_challenge', 'single': 0, 'single_score': 12.0, 'read_score': 1.0},
        ]
        threshold, eligible = joint_single_threshold(rows, 0.0)
        self.assertEqual(len(eligible), 1)
        self.assertGreater(threshold, 0.5)
        self.assertLess(threshold, 1.0)

    def test_rejects_empty_or_nonfinite_calibration(self):
        with self.assertRaises(ValueError):
            joint_single_threshold([], 0.0)
        with self.assertRaises(ValueError):
            joint_single_threshold([{'split': 'calibration', 'single': 0,
                                     'single_score': float('nan'), 'read_score': 1.0}], 0.0)
        with self.assertRaises(ValueError):
            joint_single_threshold([], float('inf'))


if __name__ == '__main__':
    unittest.main()
