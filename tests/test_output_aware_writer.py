import unittest

import torch
from scripts.deepseek_spora_output_aware_writer import (expanded_value_split, hard_negative_ids,
                                                        norm_vectors, sample_output_tokens)


class OutputAwareWriterTests(unittest.TestCase):
    def test_negatives_never_include_held_values(self):
        ids = sample_output_tokens(20, [1, 2], [3, 4], 5, 42)
        self.assertEqual(ids[:2], [1, 2])
        self.assertEqual(len(set(ids)), len(ids))
        self.assertFalse(set(ids) & {3, 4})
        self.assertEqual(ids, sample_output_tokens(20, [1, 2], [3, 4], 5, 42))
        with self.assertRaises(ValueError):
            sample_output_tokens(20, [1], [1], 5, 42)

    def test_normalization_retains_gradient(self):
        vector = torch.tensor([[3., 4.]], requires_grad=True)
        normalized = norm_vectors(vector, 2.)
        torch.testing.assert_close(normalized.norm(dim=-1), torch.tensor([2.]))
        normalized[0, 0].backward()
        self.assertTrue(torch.isfinite(vector.grad).all())
        self.assertGreater(vector.grad.abs().sum().item(), 0)

    def test_mined_tokens_exclude_held_and_existing(self):
        scores = torch.tensor([[1., 9., 8., 7., 6.], [1., 8., 9., 6., 7.]])
        self.assertEqual(hard_negative_ids(scores, [1], [2], 4), [3, 4])

    def test_expanded_training_keeps_original_holdout(self):
        train, held = expanded_value_split(list(range(500)), 350)
        self.assertEqual(held, list(range(256, 320)))
        self.assertEqual(train[:256], list(range(256)))
        self.assertEqual(train[256:], list(range(320, 414)))
        self.assertFalse(set(train) & set(held))
        filtered, same_holdout = expanded_value_split(list(range(500)), 258, lambda x: x != 320)
        self.assertEqual(filtered[-2:], [321, 322])
        self.assertEqual(same_holdout, held)
        with self.assertRaises(ValueError):
            expanded_value_split(list(range(350)), 300)


if __name__ == '__main__':
    unittest.main()
