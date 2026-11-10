import unittest

import torch

from spora.structure_transfer import SPORALayerBlend, SPORALayerReplacement, apply_position_tails


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

    def test_extend_positions_preserves_existing_outputs_and_cache(self):
        torch.manual_seed(127)
        layer = SPORALayerReplacement(16, 4, max_length=4, layer_idx=2).eval()
        values = torch.randn(1, 7, 16)
        with torch.no_grad():
            previous = layer(values[:, :4])
        old_position = layer.position.weight.detach().clone()
        layer.extend_positions(8)
        torch.testing.assert_close(layer.position.weight[:4], old_position, rtol=0, atol=0)
        torch.testing.assert_close(layer.position.weight[4:], torch.zeros_like(layer.position.weight[4:]))
        with torch.no_grad():
            preserved = layer(values[:, :4])
            extended = layer(values)
            cache = self.Cache()
            prefill = layer(values[:, :4], past_key_values=cache, use_cache=True)
            rest = [layer(values[:, i:i+1], past_key_values=cache, use_cache=True)
                    for i in range(4, 7)]
        torch.testing.assert_close(previous, preserved, rtol=0, atol=0)
        torch.testing.assert_close(torch.cat([prefill, *rest], dim=1), extended, atol=2e-6, rtol=2e-6)
        with self.assertRaisesRegex(ValueError, 'larger'):
            layer.extend_positions(8)

    def test_position_tail_restore_validates_all_before_mutating(self):
        layers = {1: SPORALayerReplacement(8, 2, max_length=4, layer_idx=1),
                  2: SPORALayerReplacement(8, 2, max_length=4, layer_idx=2)}
        sources = {'1': 'digest-one', '2': 'digest-two'}
        tails = {'1': torch.ones(4, 8), '2': torch.full((4, 8), 2.)}
        payload = {'format': 'spora-position-tail-v1', 'base_weight_sha256': 'base',
                   'sources': sources, 'old_length': 4, 'new_length': 8, 'tails': tails}
        prefixes = {i: block.position.weight.detach().clone() for i, block in layers.items()}
        bad = dict(payload, tails={'1': tails['1'], '2': torch.full((4, 8), float('nan'))})
        with self.assertRaisesRegex(ValueError, 'position tail'):
            apply_position_tails(layers, bad, 'base', sources)
        self.assertEqual([block.max_length for block in layers.values()], [4, 4])
        with self.assertRaisesRegex(ValueError, 'provenance'):
            apply_position_tails(layers, payload, 'other-base', sources)
        apply_position_tails(layers, payload, 'base', sources)
        for i, block in layers.items():
            torch.testing.assert_close(block.position.weight[:4], prefixes[i], atol=0, rtol=0)
            torch.testing.assert_close(block.position.weight[4:], tails[str(i)], atol=0, rtol=0)

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

    def test_left_padding_cannot_influence_valid_tokens(self):
        torch.manual_seed(123)
        layer = SPORALayerReplacement(16, 4, max_length=8, layer_idx=2).eval()
        values = torch.randn(2, 5, 16)
        changed = values.clone()
        changed[0, :2] = torch.randn_like(changed[0, :2]) * 20
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        positions = torch.tensor([[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]])
        with torch.no_grad():
            first = layer(values, attention_mask=mask, position_ids=positions)
            second = layer(changed, attention_mask=mask, position_ids=positions)
        torch.testing.assert_close(first[0, 2:], second[0, 2:], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(first[1], second[1])
        self.assertTrue(torch.isfinite(first).all())

    def test_cached_left_padding_matches_full_forward(self):
        torch.manual_seed(124)
        layer = SPORALayerReplacement(16, 4, max_length=8, layer_idx=2).eval()
        values = torch.randn(2, 5, 16)
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        positions = torch.tensor([[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]])
        with torch.no_grad():
            full = layer(values, attention_mask=mask, position_ids=positions)
            cache = self.Cache()
            prefill = layer(values[:, :4], attention_mask=mask[:, :4], position_ids=positions[:, :4],
                            past_key_values=cache, use_cache=True)
            last = layer(values[:, 4:], attention_mask=mask, position_ids=positions[:, 4:],
                         past_key_values=cache, use_cache=True)
        torch.testing.assert_close(prefill[0, 2:], full[0, 2:4], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(last, full[:, 4:], atol=2e-6, rtol=2e-6)

    def test_four_dimensional_additive_mask_preserves_causality(self):
        torch.manual_seed(125)
        layer = SPORALayerReplacement(16, 4, max_length=8).eval()
        values = torch.randn(1, 4, 16)
        changed = values.clone()
        changed[:, -1] += 100
        # An all-zero 4-D mask does not authorize future attention.
        additive = torch.zeros(1, 1, 4, 4)
        with torch.no_grad():
            first = layer(values, attention_mask=additive)
            second = layer(changed, attention_mask=additive)
        torch.testing.assert_close(first[:, :3], second[:, :3], atol=2e-6, rtol=2e-6)

    def test_masked_padding_has_finite_gradients_and_4d_parity(self):
        torch.manual_seed(126)
        layer = SPORALayerReplacement(16, 4, max_length=8).eval()
        values = torch.randn(1, 4, 16, requires_grad=True)
        padding = torch.tensor([[0, 0, 1, 1]])
        additive = torch.zeros(1, 1, 4, 4)
        additive[..., :2] = float('-inf')
        positions = torch.tensor([[0, 0, 0, 1]])
        padded = layer(values, attention_mask=padding, position_ids=positions)
        masked = layer(values, attention_mask=additive, position_ids=positions)
        torch.testing.assert_close(padded, masked)
        padded.square().sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in layer.parameters()))

    def test_invalid_attention_masks_are_rejected(self):
        layer = SPORALayerReplacement(16, 4, max_length=8)
        values = torch.randn(1, 3, 16)
        for mask in (torch.tensor([[1, 2, 1]]), torch.ones(1, 2), torch.ones(1, 1, 3, 2),
                     torch.full((1, 1, 3, 3), float('nan'))):
            with self.assertRaisesRegex(ValueError, 'attention mask'):
                layer(values, attention_mask=mask)

    def test_invalid_cached_mask_does_not_modify_cache(self):
        layer = SPORALayerReplacement(16, 4, max_length=8, layer_idx=2)
        cache = self.Cache()
        with self.assertRaisesRegex(ValueError, 'attention mask'):
            layer(torch.randn(1, 3, 16), attention_mask=torch.ones(1, 2),
                  past_key_values=cache, use_cache=True)
        self.assertEqual(cache.get_seq_length(2), 0)

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
