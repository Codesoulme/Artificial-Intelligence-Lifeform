import unittest

from scripts import deepseek_spora_sentence_gate_audit as audit


class SentenceGateAuditTests(unittest.TestCase):
    def test_calibration_rejects_every_observed_negative(self):
        scores = [.1, .8, .7, .95]
        labels = [0, 0, 1, 1]
        cutoff = audit.calibrated_cutoff(scores, labels)
        self.assertEqual(cutoff, .8)
        self.assertEqual([score > cutoff for score in scores], [False, False, False, True])
        with self.assertRaises(ValueError):
            audit.calibrated_cutoff([float('nan'), .2], [0, 1])
        with self.assertRaises(ValueError):
            audit.calibrated_cutoff([.2], [1])

    def test_split_is_deterministic_and_disjoint(self):
        rows = [{'group': group, 'question': f'{group}-{index}'}
                for group in ('weak_single', 'weak_multiple', 'synthetic_none') for index in range(8)]
        fit, calibration = audit.split_training(rows)
        self.assertEqual(audit.split_training(rows), (fit, calibration))
        self.assertEqual(len(fit), 18)
        self.assertEqual(len(calibration), 6)
        self.assertFalse(set(fit) & set(calibration))

    def test_synthetic_name_families_are_disjoint(self):
        names = [f'name-{index:02}' for index in range(24)]
        fit, calibration = audit.synthetic_name_groups(names)
        self.assertEqual(len(fit), 48)
        self.assertEqual(len(calibration), 24)
        self.assertFalse({row['question'] for row in fit} & {row['question'] for row in calibration})
        self.assertFalse(any(name in row['question'] for name in names[:16] for row in calibration))
        with self.assertRaises(ValueError):
            audit.synthetic_name_groups(names[:23] + [names[0]])


if __name__ == '__main__':
    unittest.main()
