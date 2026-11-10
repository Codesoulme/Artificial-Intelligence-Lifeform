import unittest

import torch

from spora.structure_transfer import SPORALayerBlend, SPORALayerReplacement


class StructureTransferTests(unittest.TestCase):
    class Cache:
        def __init__(self):
            self.values = {}

        def get_seq_length(self, layer_idx=0):
            item = self.values.get(layer_idx)
            return 0 if item is None else item[0].shape[-2]

        def update(self, key, value, layer_idx, cache_kwargs=None):
            previous = self.values.get(layer_idx)
            if previous is not None:
                key = torch.cat((previous[0], key), dim=-2)
                value = torch.cat((previous[1], value), dim=-2)
            self.values[layer_idx] = (key, value)
            return key, value

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

    def test_singleton_position_ids_broadcast_across_batch(self):
        torch.manual_seed(12)
        layer = SPORALayerReplacement(16, 4, max_length=8).eval()
        values = torch.randn(3, 5, 16)
        positions = torch.arange(5).reshape(1, -1)
        with torch.no_grad():
            broadcast = layer(values, position_ids=positions)
            explicit = layer(values, position_ids=positions.expand(3, -1))
        torch.testing.assert_close(broadcast, explicit)

    def test_rejects_cache_and_invalid_positions(self):
        layer = SPORALayerReplacement(16, 4, max_length=8)
        values = torch.randn(1, 3, 16)
        with self.assertRaisesRegex(ValueError, 'cache object'):
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

    def test_cached_prefill_and_incremental_tokens_match_full_forward(self):
        torch.manual_seed(19)
        layer = SPORALayerReplacement(16, 4, max_length=8, layer_idx=2).eval()
        values = torch.randn(1, 6, 16)
        positions = torch.arange(6).reshape(1, -1)
        with torch.no_grad():
            full = layer(values, position_ids=positions)
            cache = self.Cache()
            prefill = layer(values[:, :3], position_ids=positions[:, :3],
                            cache_position=torch.arange(3), past_key_values=cache, use_cache=True)
            incremental = [prefill]
            for index in range(3, 6):
                incremental.append(layer(values[:, index:index + 1], position_ids=positions[:, index:index + 1],
                                         cache_position=torch.tensor([index]), past_key_values=cache,
                                         use_cache=True))
        cached = torch.cat(incremental, dim=1)
        torch.testing.assert_close(cached, full, atol=2e-6, rtol=2e-6)

    def test_blend_endpoints_and_cache_guard(self):
        torch.manual_seed(42)
        teacher = torch.nn.Linear(16, 16, bias=False)
        replacement = SPORALayerReplacement(16, 4, max_length=8, layer_idx=0)
        values = torch.randn(1, 3, 16)
        reference = teacher(values)
        full = replacement(values)
        zero = SPORALayerBlend(teacher, replacement, 0.0)
        one = SPORALayerBlend(teacher, replacement, 1.0)
        torch.testing.assert_close(zero(values), reference)
        torch.testing.assert_close(one(values), full)
        with self.assertRaisesRegex(ValueError, 'cache-disabled'):
            one(values, use_cache=True)


if __name__ == '__main__':
    unittest.main()
