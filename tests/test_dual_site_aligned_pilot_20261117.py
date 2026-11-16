import unittest

import torch
from torch.nn import functional as F

from scripts.deepseek_dual_site_aligned_pilot_20261117 import dual_group_vectors
from spora.temporal import TemporalConfig, TemporalSPORA


class DualSiteAlignedTests(unittest.TestCase):
    def test_distinct_values_survive_at_both_sites_and_addresses(self):
        torch.manual_seed(78)
        model = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=16,
            key_count=8, expert_rank=4, memory_layers=(0, 1), history_writer='rls',
            current_address='orthogonal'))
        values = F.normalize(torch.randn((4, 2, 16)), dim=-1)
        observed = dual_group_vectors(model, values, torch.device('cpu'))
        self.assertEqual(observed.shape, values.shape)
        self.assertTrue(bool((F.cosine_similarity(observed, values, dim=-1) > 0.999).all()))
        self.assertTrue(bool((F.cosine_similarity(observed[:, 0], values[:, 1], dim=-1) < 0.9).all()))

    def test_rejects_wrong_dimensions_and_excess_keys(self):
        model = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=16,
            key_count=8, expert_rank=4, memory_layers=(0, 1), history_writer='rls',
            current_address='orthogonal'))
        for shape in ((2, 1, 16), (2, 2, 15), (9, 2, 16), (0, 2, 16)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                dual_group_vectors(model, torch.zeros(shape), torch.device('cpu'))
