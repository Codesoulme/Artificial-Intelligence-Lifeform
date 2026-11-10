import unittest

import torch

from scripts.deepseek_batched_preference_routing import left_pad


class BatchedPreferenceRoutingTests(unittest.TestCase):
    def test_left_padding_preserves_content_and_attention(self):
        items = [{'input_ids': torch.tensor([[2, 3, 4]]), 'attention_mask': torch.ones(1, 3, dtype=torch.long)},
                 {'input_ids': torch.tensor([[5]]), 'attention_mask': torch.ones(1, 1, dtype=torch.long)}]
        result = left_pad(items, 0)
        self.assertEqual(result['input_ids'].tolist(), [[2, 3, 4], [0, 0, 5]])
        self.assertEqual(result['attention_mask'].tolist(), [[1, 1, 1], [0, 0, 1]])

    def test_rejects_missing_inputs_or_invalid_pad(self):
        with self.assertRaises(ValueError):
            left_pad([], 0)
        with self.assertRaises(ValueError):
            left_pad([{}], None)


if __name__ == '__main__':
    unittest.main()
