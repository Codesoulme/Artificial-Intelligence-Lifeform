import unittest

from scripts.deepseek_spora_chunked_interference_20261117 import count_other_losses, summarize


class InterferenceMetricsTests(unittest.TestCase):
    def test_matches_by_key_and_excludes_updated(self):
        old = [{'key': 1, 'identity_exact': True, 'fact_exact': True, 'decode_error': None},
               {'key': 2, 'identity_exact': True, 'fact_exact': True, 'decode_error': None},
               {'key': 3, 'identity_exact': False, 'fact_exact': False, 'decode_error': 'bad'}]
        new = [{'key': 3, 'identity_exact': True, 'fact_exact': True, 'decode_error': None},
               {'key': 1, 'identity_exact': False, 'fact_exact': False, 'decode_error': 'bad'},
               {'key': 2, 'identity_exact': False, 'fact_exact': True, 'decode_error': None}]
        self.assertEqual(count_other_losses(old, new, 1), 1)
        self.assertEqual(summarize(old)['both_exact'], 2)
        self.assertEqual(summarize(old)['decode_errors'], 1)
        with self.assertRaises(ValueError):
            count_other_losses(old, new[:-1], 1)
        with self.assertRaises(ValueError):
            count_other_losses(old, [new[0], new[0], new[1]], 1)
