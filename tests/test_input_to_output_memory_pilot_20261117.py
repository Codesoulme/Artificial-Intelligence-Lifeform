import unittest

import torch

from scripts.deepseek_input_to_output_memory_pilot_20261117 import fit_projection


class InputOutputProjectionTests(unittest.TestCase):
    def test_fit_predicts_heldout_linear_mapping(self):
        torch.manual_seed(31)
        x = torch.randn(100, 5)
        target = torch.randn(5, 7)
        fitted = fit_projection(x[:80], x[:80] @ target, 0.001)
        torch.testing.assert_close(x[80:] @ fitted, x[80:] @ target, rtol=1e-3, atol=1e-3)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            fit_projection(torch.ones(3, 2), torch.ones(2, 2), 0.1)
        with self.assertRaises(ValueError):
            fit_projection(torch.ones(3, 2), torch.ones(3, 2), 0)
        with self.assertRaises(ValueError):
            fit_projection(torch.tensor([[float('nan')], [0.]]), torch.ones(2, 2), 0.1)
