import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

import torch

try:
    import transformers
except ModuleNotFoundError:
    transformers = None

if transformers is not None:
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Config, Qwen2RotaryEmbedding

    from spora.pretrained_attention import (AbsoluteAttention, SPORAFusedAttention, SplitPositionAttention,
                                            ValuePositionAttention, load_value_position_attention)
    from spora.research_partial_query_key import PartialQueryKeyPositionAttention, select_query_key_rows
    from spora.research_recurrent_decode import project_with_history
    from spora.value_position_checkpoint import decode_query_key_positions, query_key_position_bundle


@unittest.skipIf(transformers is None, 'requires the optional Transformers runtime')
class SplitPositionAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.config = Qwen2Config(hidden_size=24, num_attention_heads=6, num_key_value_heads=2,
                                  num_hidden_layers=1, intermediate_size=56)
        self.config._attn_implementation = 'sdpa'
        native = Qwen2Attention(self.config, 0).eval()
        self.rotary = SPORAFusedAttention(native, keep_groups=True).eval()
        self.absolute = AbsoluteAttention(native, 12, keep_groups=True).eval()
        with torch.no_grad():
            self.absolute.position.weight.normal_()
        self.hidden = torch.randn(1, 5, 24)
        self.positions = torch.arange(5)[None]
        self.embeddings = Qwen2RotaryEmbedding(self.config)(self.hidden, self.positions)

    def test_recurrent_history_uses_unrotated_keys_without_changing_local_projection(self):
        shifted = self.positions + 21
        moved = Qwen2RotaryEmbedding(self.config)(self.hidden, shifted)
        local, history = project_with_history(self.rotary, self.hidden, self.embeddings, self.positions, 'unrotated')
        other_local, other_history = project_with_history(self.rotary, self.hidden, moved, shifted, 'unrotated')
        for actual, expected in zip(local, self.rotary.project(self.hidden, self.embeddings, self.positions), strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for actual, expected in zip(history, other_history, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertFalse(torch.equal(local[1], other_local[1]))

    def test_only_selected_queries_and_keys_use_absolute_position(self):
        args = self.hidden, self.embeddings, self.positions
        rotary = self.rotary.project(*args)
        absolute = self.absolute.project(*args)
        for index in (0, 1):
            with self.subTest(group=index):
                result = SplitPositionAttention(self.rotary, self.absolute, (index,)).project(*args)
                for original, migrated, actual, width in zip(rotary, absolute, result, (3, 1, 1), strict=True):
                    start = index * width
                    stop = start + width
                    torch.testing.assert_close(actual[:, start:stop], migrated[:, start:stop], rtol=0, atol=0)
                    other = (1 - index) * width
                    torch.testing.assert_close(actual[:, other:other + width],
                                               original[:, other:other + width], rtol=0, atol=0)

    def test_one_cache_keeps_original_number_of_key_value_heads(self):
        mixed = SplitPositionAttention(self.rotary, self.absolute, (1,)).eval()
        cache = DynamicCache(config=self.config)
        with torch.inference_mode():
            expected = mixed(self.hidden, self.embeddings, None, position_ids=self.positions)[0][:, -1]
            for start, stop in ((0, 4), (4, 5)):
                position = self.positions[:, start:stop]
                actual = mixed(self.hidden[:, start:stop],
                               tuple(value[:, start:stop] for value in self.embeddings), None,
                               past_key_values=cache, position_ids=position, cache_position=position[0])[0]
        torch.testing.assert_close(actual[:, -1], expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(cache.layers[0].keys.shape, (1, 2, 5, 4))
        self.assertEqual(cache.layers[0].values.shape, (1, 2, 5, 4))

    def test_query_key_only_and_value_only_change_disjoint_projections(self):
        args = self.hidden, self.embeddings, self.positions
        expected = self.rotary.project(*args)
        both = SplitPositionAttention(self.rotary, self.absolute, (0,)).project(*args)
        key_only = SplitPositionAttention(self.rotary, self.absolute, (0,), value_groups=()).project(*args)
        value_only = SplitPositionAttention(self.rotary, self.absolute, (), value_groups=(0,)).project(*args)
        for projections, reference in (
            (key_only, (*both[:2], expected[2])),
            (value_only, (*expected[:2], both[2])),
        ):
            for actual, wanted in zip(projections, reference, strict=True):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

    def test_saved_value_component_restores_only_one_head(self):
        args = self.hidden, self.embeddings, self.positions
        group = 1
        start = sum(self.absolute.sizes[:2]) + group * self.absolute.head_dim
        stop = start + self.absolute.head_dim
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'value-position.pt'
            torch.save(dict(format='spora-value-position-component-v1', layer=0, group=group,
                            source_sha256='base', position=self.absolute.position.weight.detach(),
                            weight=self.absolute.qkv.weight[start:stop].detach(),
                            bias=self.absolute.qkv.bias[start:stop].detach()), path)
            digest = sha256(path.read_bytes()).hexdigest()
            component = load_value_position_attention(path, self.rotary, checkpoint_digest='base',
                                                      component_digest=digest)
            with self.assertRaisesRegex(ValueError, 'different layer or checkpoint'):
                load_value_position_attention(path, self.rotary, checkpoint_digest='other',
                                              component_digest=digest)
        expected = SplitPositionAttention(self.rotary, self.absolute, (), value_groups=(group,)).project(*args)
        for actual, wanted in zip(component.project(*args), expected, strict=True):
            torch.testing.assert_close(actual, wanted)

    def test_partial_query_key_uses_both_halves_and_keeps_the_value(self):
        state = self.absolute.state_dict()
        group = 1
        start = sum(self.absolute.sizes[:2]) + group * self.absolute.head_dim
        stop = start + self.absolute.head_dim
        value = ValuePositionAttention(self.rotary, group, state['position.weight'],
                                       state['qkv.weight'][start:stop], state['qkv.bias'][start:stop])
        pairs = (1,)
        weight, bias = select_query_key_rows(value, pairs, state)
        mixed = PartialQueryKeyPositionAttention(value, pairs, weight, bias).eval()
        args = self.hidden, self.embeddings, self.positions
        expected = list(value.project(*args))
        projected = self.absolute.project(*args)
        for index, (head_start, width) in enumerate(((3, 3), (1, 1))):
            expected[index] = expected[index].clone()
            section = slice(head_start, head_start + width)
            for coordinate in (1, 3):
                expected[index][:, section, :, coordinate] = projected[index][:, section, :, coordinate]
        for actual, wanted in zip(mixed.project(*args), expected, strict=True):
            torch.testing.assert_close(actual, wanted)
        cache = DynamicCache(config=self.config)
        with torch.inference_mode():
            full = mixed(*args[:2], None, position_ids=self.positions)[0][:, -1]
            for start, stop in ((0, 4), (4, 5)):
                pos = self.positions[:, start:stop]
                chunk = mixed(self.hidden[:, start:stop], tuple(x[:, start:stop] for x in self.embeddings),
                              None, past_key_values=cache, position_ids=pos, cache_position=pos[0])[0]
        torch.testing.assert_close(chunk[:, -1], full, rtol=1e-5, atol=1e-6)
        self.assertEqual(cache.layers[0].keys.shape, (1, 2, 5, 4))

    def test_partial_query_key_groups_compose_without_overwriting_each_other(self):
        state = self.absolute.state_dict()
        start = sum(self.absolute.sizes[:2])
        stop = start + self.absolute.head_dim
        values = ValuePositionAttention(self.rotary, 0, state['position.weight'],
                                        state['qkv.weight'][start:stop], state['qkv.bias'][start:stop])
        first_weight, first_bias = select_query_key_rows(values, (1,), state)
        second_weight, second_bias = select_query_key_rows(values, (0,), state, group=1)
        first = PartialQueryKeyPositionAttention(values, (1,), first_weight, first_bias)
        combined = PartialQueryKeyPositionAttention(first, (0,), second_weight, second_bias, group=1)
        second = PartialQueryKeyPositionAttention(values, (0,), second_weight, second_bias, group=1)
        args = self.hidden, self.embeddings, self.positions
        first_q, first_k, first_v = first.project(*args)
        second_q, second_k, _ = second.project(*args)
        query, key, value = combined.project(*args)
        torch.testing.assert_close(query, torch.cat((first_q[:, :3], second_q[:, 3:]), dim=1))
        torch.testing.assert_close(key, torch.cat((first_k[:, :1], second_k[:, 1:]), dim=1))
        torch.testing.assert_close(value, first_v)
        with self.assertRaisesRegex(ValueError, 'only be migrated once'):
            PartialQueryKeyPositionAttention(first, (1,), first_weight, first_bias)
        config = self.config.to_dict()
        specs, tensors = query_key_position_bundle({0: combined}, {0: values}, config)
        self.assertEqual(specs, {'0': {'0': [1], '1': [0]}})
        decoded_values = {0: (values.group, values.position.weight, values.weight, values.bias)}
        decoded = decode_query_key_positions(specs, tensors, decoded_values, config)
        self.assertEqual([(group, pairs) for group, pairs, _, _ in decoded[0]], [(0, [1]), (1, [0])])
        restored = values
        for group, pairs, weight, bias in decoded[0]:
            restored = PartialQueryKeyPositionAttention(restored, pairs, weight, bias, group=group)
        for actual, expected in zip(restored.project(*args), (query, key, value), strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'group metadata'):
            decode_query_key_positions({'0': {'2': [0]}}, tensors, decoded_values, config)
        broken = dict(tensors)
        del broken['query_key_position/0/1/bias']
        with self.assertRaisesRegex(ValueError, 'Missing or unexpected'):
            decode_query_key_positions(specs, broken, decoded_values, config)
