import unittest

import torch

from spora.memory_output import MemoryOutputAdapter, MemoryOutputLogitsProcessor


class MemoryOutputTests(unittest.TestCase):
    def test_zero_initialized_adapter_is_inactive_and_finite(self):
        torch.manual_seed(4)
        adapter = MemoryOutputAdapter(17, 101, rank=8)
        logits = torch.randn(2, 17)
        bias = adapter(logits)
        self.assertEqual(tuple(bias.shape), (2, 101))
        self.assertTrue(torch.isfinite(bias).all())
        self.assertTrue(torch.equal(bias, torch.zeros_like(bias)))

    def test_processor_preserves_batch_and_rejects_wrong_width(self):
        adapter = MemoryOutputAdapter(4, 9, rank=3)
        with torch.no_grad():
            adapter.rank_to_vocab.weight[0, 0] = 1
        processor = MemoryOutputLogitsProcessor(adapter, torch.tensor([[4., 0., 0., 0.]]))
        scores = torch.zeros(2, 9)
        result = processor(torch.zeros(2, 1, dtype=torch.long), scores)
        self.assertEqual(tuple(result.shape), (2, 9))
        self.assertTrue(torch.isfinite(result).all())
        second = processor(torch.zeros(2, 2, dtype=torch.long), scores)
        self.assertTrue(torch.equal(second, scores))
        boosted = MemoryOutputLogitsProcessor(adapter, torch.tensor([[4., 0., 0., 0.]]), gain=2.)
        boosted_result = boosted(torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 9))
        self.assertTrue(torch.equal(boosted_result, result[:1] * 2))
        with self.assertRaises(ValueError):
            processor(torch.zeros(2, 1, dtype=torch.long), torch.zeros(2, 8))

    def test_config_roundtrip_values(self):
        adapter = MemoryOutputAdapter(7, 13, rank=5, max_scale=3.)
        self.assertEqual(adapter.config_dict(), {'value_count': 7, 'vocab_size': 13,
                                                  'rank': 5, 'max_scale': 3.})


if __name__ == '__main__':
    unittest.main()
