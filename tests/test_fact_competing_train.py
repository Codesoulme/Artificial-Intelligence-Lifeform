import unittest
from collections import Counter
from scripts.deepseek_fact_competing_train_20261117 import build_groups, TRAIN_SAME


class CompetingFactsTests(unittest.TestCase):
    def test_training_reuses_same_values_as_known_and_unknown(self):
        groups = build_groups()
        counts = Counter((row['fact'], row['kind']) for row in groups['train'])
        for fact in TRAIN_SAME:
            self.assertGreater(counts[fact, 'known'], 0)
            self.assertGreater(counts[fact, 'unknown'], 0)
        training = {row['text'] for row in groups['train']}
        for name, rows in groups.items():
            if name != 'train':
                self.assertFalse(training & {row['text'] for row in rows})
