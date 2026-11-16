import unittest

import torch
from torch.nn import functional as F

from scripts.deepseek_output_aligned_holdout_20261117 import choose_unseen, grouped_vectors, metrics
from spora.temporal import TemporalConfig, TemporalSPORA


class OutputAlignedHoldoutTests(unittest.TestCase):
    def test_selection_is_corpus_disjoint_and_deterministic(self):
        train = {1: 3, 2: 5}
        test = {1: 9, 2: 4, 3: 1, 4: 2, 5: 1, 6: 1}
        selected, eligible = choose_unseen(train, test, 3, 17)
        self.assertEqual((selected, eligible), choose_unseen(train, test, 3, 17))
        self.assertTrue(set(selected).isdisjoint(train))
        self.assertEqual(eligible, 4)
        with self.assertRaises(ValueError):
            choose_unseen(train, test, 5, 17)

    def test_grouped_writes_recover_distinct_vectors(self):
        torch.manual_seed(44)
        model = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=16,
                     key_count=8, expert_rank=4, memory_layers=(0, 1), history_writer='rls',
                     current_address='orthogonal'))
        values = F.normalize(torch.randn((4, 16)), dim=-1)
        observed = grouped_vectors(model, values, torch.device('cpu'))
        self.assertTrue(bool((F.cosine_similarity(observed, values, dim=-1) > 0.999).all()))

    def test_pair_metrics_does_not_credit_changed_wrong_outputs(self):
        result = metrics([1, 3, 2, 1], [1, 2, 3, 4])
        self.assertEqual(result['exact'], 1)
        self.assertEqual(result['pair_both_correct'], 0)
        self.assertEqual(result['pair_outputs_differ'], 2)
