import unittest
from scripts.deepseek_cmrc_reading_train_pilot_20261117 import choose_replay


class ReadingReplayTests(unittest.TestCase):
    def test_replay_balanced_and_deterministic(self):
        rows = [{'kind': kind, 'id': i} for kind in ('known', 'unknown') for i in range(20)]
        first = choose_replay(rows)
        self.assertEqual(first, choose_replay(rows))
        self.assertEqual(sum(r['kind'] == 'known' for r in first), 16)
        self.assertEqual(sum(r['kind'] == 'unknown' for r in first), 16)
