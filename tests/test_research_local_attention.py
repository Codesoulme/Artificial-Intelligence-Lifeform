import unittest

import torch

try:
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from transformers.cache_utils import DynamicCache
except ModuleNotFoundError:
    Qwen2Config = None

from spora.integrated_open_base import IntegratedOpenBase, NativeSPORAMemoryReader
from spora.research_local_attention import local_attention


@unittest.skipIf(Qwen2Config is None, 'requires the optional Transformers runtime')
class LocalAttentionTests(unittest.TestCase):
    @torch.no_grad()
    def test_local_window_limits_selected_cache_and_matches_uncached_logits(self):
        torch.manual_seed(11)
        config = Qwen2Config(vocab_size=19, hidden_size=16, intermediate_size=24, num_hidden_layers=3,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32)
        config._attn_implementation = 'sdpa'
        base = Qwen2ForCausalLM(config).eval()
        decoder = IntegratedOpenBase(base, NativeSPORAMemoryReader(16, 8, .5), (0, 2)).eval()
        decoder.migrate_pretrained_blocks()
        tokens = torch.tensor([[3, 5, 6, 4, 2, 9, 7, 8, 2]])
        short = decoder(tokens[:, :3], use_cache=False).logits
        full = decoder(tokens, use_cache=False).logits
        with local_attention(decoder, (1,), 4):
            torch.testing.assert_close(decoder(tokens[:, :3], use_cache=False).logits, short)
            expected = decoder(tokens, use_cache=False).logits
            self.assertGreater(float((full[:, -1] - expected[:, -1]).abs().max()), .0001)
            cache = DynamicCache(config=config)
            decoder(tokens[:, :7], past_key_values=cache, use_cache=True)
            for stop in (8, 9):
                actual = decoder(tokens[:, stop - 1:stop], past_key_values=cache, use_cache=True).logits
                torch.testing.assert_close(actual[:, -1], expected[:, stop - 1], atol=1e-6, rtol=1e-5)
            self.assertEqual(cache.layers[0].keys.shape[-2], 9)
            self.assertEqual(cache.layers[1].keys.shape[-2], 3)
            self.assertEqual(cache.layers[2].keys.shape[-2], 9)
        torch.testing.assert_close(decoder(tokens, use_cache=False).logits, full, rtol=0, atol=0)
