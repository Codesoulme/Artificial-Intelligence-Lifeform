import unittest

import torch

from spora.continuous_text_memory import ContinuousTextCodec


class ContinuousTextMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        embedding = torch.randn(16, 8)
        self.codec = ContinuousTextCodec(embedding, slot_count=4, seed=17, eos_token_id=15)

    def test_round_trip_single_and_multitoken_sequences(self):
        for tokens in ([2], [3, 4], [5, 6, 7]):
            decoded = self.codec.decode(self.codec.encode(torch.tensor(tokens, dtype=torch.long)))
            self.assertEqual(decoded['token_ids'], tokens + [15])
            self.assertEqual(len(decoded['margins']), len(tokens) + 1)

    def test_zero_slots_decode_as_empty(self):
        decoded = self.codec.decode(torch.zeros(self.codec.slot_count * self.codec.slot_width, dtype=torch.float32))
        self.assertEqual(decoded['token_ids'], [])
        self.assertEqual(decoded['margins'], [])

    def test_capacity_and_vocab_validation(self):
        with self.assertRaises(ValueError):
            self.codec.encode(torch.tensor([1, 2, 3, 4], dtype=torch.long))
        with self.assertRaises(ValueError):
            self.codec.encode(torch.tensor([1, 99], dtype=torch.long))

    def test_can_target_a_narrower_memory_width(self):
        codec = ContinuousTextCodec(torch.randn(16, 8), memory_width=4, slot_count=2,
                                    seed=23, eos_token_id=15)
        self.assertEqual(codec.slot_width, 2)
        self.assertEqual(codec.encode(torch.tensor([2], dtype=torch.long)).numel(), 4)


if __name__ == '__main__':
    unittest.main()
