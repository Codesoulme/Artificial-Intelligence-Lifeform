import unittest

import torch

from spora.memory_output import MemoryOutputAdapter, MemoryOutputLogitsProcessor, MemorySequenceLogitsProcessor


class MemoryOutputTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required to test cross-device generation')
    def test_processor_accepts_cpu_memory_readout_with_cuda_adapter(self):
        adapter = MemoryOutputAdapter(4, 9, rank=3).cuda()
        with torch.no_grad():
            adapter.rank_to_vocab.weight.normal_()
        memory_logits = torch.tensor([[4., 0., 0., 0.]])
        scores = torch.zeros(1, 9, device='cuda')
        processor = MemoryOutputLogitsProcessor(adapter, memory_logits)
        actual = processor(torch.zeros(1, 2, dtype=torch.long, device='cuda'), scores)
        torch.testing.assert_close(actual, adapter(memory_logits.cuda()))

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

    def test_sequence_bias_uses_prefix_offset_and_leaves_later_tokens_unchanged(self):
        bias = torch.tensor([[0., 2., 0.], [3., 0., 0.]])
        processor = MemorySequenceLogitsProcessor(bias, prefix_length=4)
        bias.zero_()
        scores = torch.ones(2, 3, dtype=torch.float16)
        for length, expected in ((4, [1., 3., 1.]), (5, [4., 1., 1.]), (6, [1., 1., 1.]), (4, [1., 3., 1.])):
            with self.subTest(length=length):
                result = processor(torch.zeros(2, length, dtype=torch.long), scores)
                torch.testing.assert_close(result, scores.new_tensor([expected, expected]))

    def test_sequence_bias_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            MemorySequenceLogitsProcessor(torch.tensor([[float('inf')]]), prefix_length=1)
        with self.assertRaises(ValueError):
            MemorySequenceLogitsProcessor(torch.zeros(1, 3), prefix_length=0)
        processor = MemorySequenceLogitsProcessor(torch.zeros(1, 3), prefix_length=2)
        with self.assertRaises(ValueError):
            processor(torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 3))
        with self.assertRaises(ValueError):
            processor(torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 4))


if __name__ == '__main__':
    unittest.main()
