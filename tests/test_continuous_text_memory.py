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

    def test_optional_minimum_norm_stops_near_zero_residual(self):
        code = torch.zeros(self.codec.slot_count * self.codec.slot_width)
        code[:self.codec.slot_width] = 1e-9
        self.assertNotEqual(self.codec.decode(code)['token_ids'], [])
        self.assertEqual(self.codec.decode(code, min_slot_norm=1e-6)['token_ids'], [])

    def test_minimum_norm_stops_at_first_small_slot(self):
        code = self.codec.encode(torch.tensor([2, 3], dtype=torch.long))
        start = self.codec.slot_width
        code[start:2 * start] *= 1e-9
        self.assertEqual(self.codec.decode(code, min_slot_norm=1e-6)['token_ids'], [2])

    def test_minimum_norm_rejects_invalid_thresholds(self):
        code = torch.zeros(self.codec.slot_count * self.codec.slot_width)
        for value in (True, -1, float('nan'), float('inf'), '0.1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.codec.decode(code, min_slot_norm=value)

    def test_capacity_and_vocab_validation(self):
        with self.assertRaises(ValueError):
            self.codec.encode(torch.tensor([1, 2, 3, 4], dtype=torch.long))
        with self.assertRaises(ValueError):
            self.codec.encode(torch.tensor([1, 99], dtype=torch.long))
        with self.assertRaises(ValueError):
            self.codec.encode(torch.tensor([1, 15, 2], dtype=torch.long))

    def test_zero_embedding_cannot_be_encoded_as_an_empty_slot(self):
        embedding = torch.randn(16, 8)
        embedding[3].zero_()
        codec = ContinuousTextCodec(embedding, slot_count=4, eos_token_id=15)
        with self.assertRaises(ValueError):
            codec.encode(torch.tensor([3], dtype=torch.long))

    def test_nonfinite_embedding_is_rejected(self):
        embedding = torch.randn(16, 8)
        embedding[1, 0] = float('nan')
        with self.assertRaises(ValueError):
            ContinuousTextCodec(embedding, slot_count=4)

    def test_can_target_a_narrower_memory_width(self):
        codec = ContinuousTextCodec(torch.randn(16, 8), memory_width=4, slot_count=2,
                                    seed=23, eos_token_id=15)
        self.assertEqual(codec.slot_width, 2)
        self.assertEqual(codec.encode(torch.tensor([2], dtype=torch.long)).numel(), 4)


if __name__ == '__main__':
    unittest.main()
