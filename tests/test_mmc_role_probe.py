import unittest

from scripts.deepseek_mmc_role_probe import role_features


class RoleFeatureTests(unittest.TestCase):
    def test_encodes_heuristic_and_speaker_group(self):
        rows = [{'heuristic': h, 'speaker_count': count}
                for count in (1, 2) for h in (0, 1)]
        values = role_features(rows)
        self.assertEqual(values.tolist(), [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    def test_rejects_invalid_inputs(self):
        for row in ({'heuristic': True, 'speaker_count': 2}, {'heuristic': 2, 'speaker_count': 2},
                    {'heuristic': 0, 'speaker_count': 0}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                role_features([row])


if __name__ == '__main__':
    unittest.main()
