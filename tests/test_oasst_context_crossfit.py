import unittest
from scripts.train_oasst_context_crossfit_20261118 import metrics


class CrossfitMetricTests(unittest.TestCase):
    def test_triplet_is_not_passed_by_perpetual_retrieval(self):
        rows = [{'case': 0, 'history_dependent': 1, 'variant': v, 'need': n}
                for v, n in [('current_only', 1), ('original_history', 0), ('wrong_history', 1)]]
        self.assertEqual(metrics(rows, [1, 1, 1])['dependent_all_three_correct'], 0)
        self.assertEqual(metrics(rows, [1, 0, 1])['dependent_all_three_correct'], 1)
        self.assertEqual(metrics(rows, [0, 0, 0])['wrong_history']['fn'], 1)
