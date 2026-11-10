import unittest

import torch

from spora.structure_transfer import SPORALayerReplacement


class StructureTransferTests(unittest.TestCase):
    def test_full_sequence_is_causal_and_shape_preserving(self):
        torch.manual_seed(8)
        layer = SPORALayerReplacement(16, 4, max_length=8).eval()
        values = torch.randn(2, 5, 16)
        changed = values.clone()
        changed[:, 4] += 100
        positions = torch.arange(5).expand(2, -1)
        with torch.no_grad():
            first = layer(values, position_ids=positions)
            second = layer(changed, position_ids=positions)
        self.assertEqual(first.shape, values.shape)
        torch.testing.assert_close(first[:, :4], second[:, :4])
        self.assertFalse(torch.equal(first[:, 4], second[:, 4]))

    def test_rejects_cache_and_invalid_positions(self):
        layer = SPORALayerReplacement(16, 4, max_length=8)
        values = torch.randn(1, 3, 16)
        with self.assertRaisesRegex(ValueError, 'KV cache'):
            layer(values, use_cache=True)
        with self.assertRaisesRegex(ValueError, 'position IDs'):
            layer(values, position_ids=torch.tensor([[0, 1, 8]]))

    def test_output_returns_to_teacher_dtype(self):
        layer = SPORALayerReplacement(16, 4, max_length=8).half().eval()
        values = torch.randn(1, 4, 16, dtype=torch.float16)
        with torch.no_grad():
            output = layer(values)
        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
