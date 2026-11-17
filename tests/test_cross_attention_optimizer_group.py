import unittest
import torch
from scripts.deepseek_spora_semantic_bridge_pilot import cross_attention_optimizer_group


class CrossAttentionOptimizerGroupTests(unittest.TestCase):
    def test_assigns_only_cross_parameters_and_preserves_other_groups(self):
        main, cross, separate = (torch.nn.Parameter(torch.tensor(1.)) for _ in range(3))
        groups = [{'params': [main, cross]}, {'params': [separate], 'lr': 0.02}]
        result = cross_attention_optimizer_group(groups, [cross], 0.003)
        self.assertEqual([id(p) for p in result[0]['params']], [id(main)])
        self.assertEqual([id(p) for p in result[1]['params']], [id(separate)])
        self.assertEqual(result[1]['lr'], 0.02)
        self.assertEqual([id(p) for p in result[2]['params']], [id(cross)])
        self.assertEqual(result[2]['lr'], 0.003)
        self.assertEqual(len(groups[0]['params']), 2)

    def test_rejects_duplicate_or_unregistered_parameters(self):
        p, q = torch.nn.Parameter(torch.tensor(1.)), torch.nn.Parameter(torch.tensor(2.))
        for selected, lr in (([p, p], 0.1), ([q], 0.1), ([p], 0), ([p], float('nan'))):
            with self.subTest(lr=lr, count=len(selected)), self.assertRaises(ValueError):
                cross_attention_optimizer_group([{'params': [p]}], selected, lr)
