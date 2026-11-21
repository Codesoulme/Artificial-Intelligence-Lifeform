import unittest

import torch

try:
    import transformers
except ModuleNotFoundError:
    transformers = None

if transformers is not None:
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Config, Qwen2RotaryEmbedding

    from spora.pretrained_attention import AbsoluteAttention, SPORAFusedAttention, SplitPositionAttention


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
