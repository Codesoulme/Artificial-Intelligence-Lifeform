import unittest

import torch

from scripts.deepseek_named_incremental_probe import ridge_statistics, ridge_update


class NamedIncrementalTests(unittest.TestCase):
    def test_incremental_matches_batch_without_mutating_parent(self):
        torch.manual_seed(18)
        x = torch.randn(9, 2, 5)
        y = torch.randn(9, 2, 3)
        weights, precision = ridge_statistics(x[:8], y[:8])
        old_weights, old_precision = weights.clone(), precision.clone()
        actual_w, actual_p = ridge_update(weights, precision, x[8], y[8])
        expected_w, expected_p = ridge_statistics(x, y)
        self.assertTrue(torch.allclose(actual_w, expected_w, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(actual_p, expected_p, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.equal(weights, old_weights))
        self.assertTrue(torch.equal(precision, old_precision))

    def test_bad_inputs_are_rejected(self):
        weights, precision = ridge_statistics(torch.ones(2, 2, 3), torch.ones(2, 2, 4))
        with self.assertRaises(ValueError):
            ridge_update(weights, precision, torch.full((2, 3), float('nan')), torch.ones(2, 4))
        with self.assertRaises(ValueError):
            ridge_statistics(torch.ones(2, 2, 3), torch.ones(3, 2, 4))


if __name__ == '__main__':
    unittest.main()
