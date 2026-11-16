import unittest

from scripts.deepseek_spora_chunked_history_20261117 import summarize


class HistoryMetricsTests(unittest.TestCase):
    def test_partial_failures_and_decode_errors(self):
        rows = [{'identity_exact': True, 'fact_exact': True, 'decode_error': None},
                {'identity_exact': True, 'fact_exact': False, 'decode_error': None},
                {'identity_exact': False, 'fact_exact': False, 'decode_error': 'bad'}]
        self.assertEqual(summarize(rows), {'packets': 3, 'identity_exact': 2, 'fact_exact': 1,
                                          'both_exact': 1, 'decode_errors': 1})
