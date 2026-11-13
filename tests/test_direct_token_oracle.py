import unittest

import torch

from scripts.deepseek_spora_direct_token_oracle import DirectTokenOracle


class DirectTokenOracleTests(unittest.TestCase):
    def test_biases_target_token_at_each_answer_step(self):
        processor = DirectTokenOracle((2, 4), prompt_length=3, bias=7.)
        scores = torch.zeros(1, 6)
        first = processor(torch.zeros(1, 3, dtype=torch.long), scores)
        second = processor(torch.zeros(1, 4, dtype=torch.long), scores)
        after = processor(torch.zeros(1, 5, dtype=torch.long), scores)
        self.assertEqual(int(first.argmax()), 2)
        self.assertEqual(int(second.argmax()), 4)
        self.assertEqual(int(after.argmax()), 0)

    def test_does_not_mutate_input_scores(self):
        processor = DirectTokenOracle((3,), prompt_length=2, bias=5.)
        scores = torch.zeros(1, 5)
        processor(torch.zeros(1, 2, dtype=torch.long), scores)
        self.assertTrue(torch.equal(scores, torch.zeros_like(scores)))


if __name__ == '__main__':
    unittest.main()
