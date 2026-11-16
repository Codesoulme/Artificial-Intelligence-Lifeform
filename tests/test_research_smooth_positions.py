import copy
import unittest

import torch

from spora.research_smooth_positions import SmoothPositionTail
from spora.structure_transfer import SPORALayerReplacement


class SmoothPositionTailTests(unittest.TestCase):
    def test_zero_tail_preserves_prefix_and_trains_shared_coefficients(self):
        prefix = torch.randn(8, 4)
        module = SmoothPositionTail(prefix, 16)
        positions = torch.tensor([[0, 7, 8, 15]])
        result = module(positions)
        torch.testing.assert_close(result[0, :2], prefix[[0, 7]])
        torch.testing.assert_close(result[0, 2:], torch.zeros(2, 4))
        result.sum().backward()
        self.assertIsNotNone(module.coefficients.grad)
        self.assertGreater(float(module.coefficients.grad.abs().sum()), 0.)
        self.assertFalse(module.prefix.requires_grad)

    def test_materialized_table_matches_replacement_forward(self):
        torch.manual_seed(12)
        block = SPORALayerReplacement(width=8, heads=2, max_length=8, layer_idx=0).eval()
        block.extend_positions(16)
        reference = copy.deepcopy(block)
        smooth = SmoothPositionTail(block.position.weight[:8], 16)
        with torch.no_grad():
            smooth.coefficients.normal_(std=.1)
            reference.position.weight.copy_(smooth.weight)
        block.position = smooth
        inputs = torch.randn(1, 12, 8)
        torch.testing.assert_close(block(inputs), reference(inputs))

    def test_invalid_prefix_or_length_rejected(self):
        with self.assertRaises(ValueError):
            SmoothPositionTail(torch.zeros(4, 8), 4)
        with self.assertRaises(ValueError):
            SmoothPositionTail(torch.full((4, 8), float('nan')), 8)


if __name__ == '__main__':
    unittest.main()
