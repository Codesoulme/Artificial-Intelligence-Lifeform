import unittest
from scripts.deepseek_evidence_pooling_ablation_20261118 import pooled, evaluate, torch


class PoolingTests(unittest.TestCase):
    def test_padding_never_enters_features(self):
        x = torch.tensor([[[2., 4.], [6., 8.], [999., 999.]],
                          [[999., 999.], [2., 4.], [6., 8.]]])
        result = pooled(x, torch.tensor([[1, 1, 0], [0, 1, 1]]))
        self.assertTrue(torch.equal(result['mean'], torch.tensor([[4., 6.], [4., 6.]])))
        self.assertTrue(torch.equal(result['last'], torch.tensor([[6., 8.], [6., 8.]])))

    def test_correct_ranking_can_hide_no_answer_failure(self):
        rows = [{'pair': 0, 'need': y, 'condition': str(i)} for i, y in enumerate([0, 1, 1, 1])]
        result = evaluate(rows, [-3., -1., 1., 2.], 0., True)['candidate_sets']
        self.assertEqual(result['accepted_right'], 1)
        self.assertEqual(result['accepted_when_all_wrong'], 1)
        self.assertEqual(result['all_correct'], 0)

    def test_empty_sequence_rejected(self):
        with self.assertRaises(ValueError):
            pooled(torch.zeros(1, 2, 3), torch.zeros(1, 2))
