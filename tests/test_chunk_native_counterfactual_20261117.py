import unittest

import torch

from scripts.deepseek_chunk_native_counterfactual_20261117 import replace_query_value


class CounterfactualMatrixTests(unittest.TestCase):
    def test_replaces_one_read_and_preserves_orthogonal_read(self):
        matrix = torch.tensor([[[1., 2.], [3., 4.]]])
        original = matrix.clone()
        address = torch.tensor([[2., 0.]])
        target = torch.tensor([[7., 9.]])
        changed = replace_query_value(matrix, address, target)
        torch.testing.assert_close(changed @ torch.tensor([[[1.], [0.]]]), target.unsqueeze(-1))
        torch.testing.assert_close(changed @ torch.tensor([[[0.], [1.]]]), matrix @ torch.tensor([[[0.], [1.]]]))
        torch.testing.assert_close(matrix, original)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            replace_query_value(torch.eye(2).unsqueeze(0), torch.zeros((1, 2)), torch.ones((1, 2)))
