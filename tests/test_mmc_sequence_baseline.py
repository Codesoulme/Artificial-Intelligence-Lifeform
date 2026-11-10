import unittest

import torch

from scripts.deepseek_mmc_sequence_baseline import make_prompt, sequence_logprob


class SequenceBaselineTests(unittest.TestCase):
    def test_sums_conditional_answer_token_probabilities(self):
        logits = torch.tensor([[0., 0., 0.], [0., 1., 2.], [2., 0., 1.], [0., 0., 0.]])
        ids = torch.tensor([0, 1, 2, 0])
        result = sequence_logprob(logits, ids, 2)
        expected = logits[1].log_softmax(-1)[2] + logits[2].log_softmax(-1)[0]
        self.assertAlmostEqual(result, float(expected), places=6)

    def test_rejects_invalid_shapes_and_nonfinite_scores(self):
        with self.assertRaises(ValueError):
            sequence_logprob(torch.zeros(3, 2), torch.tensor([0, 1]), 1)
        with self.assertRaises(ValueError):
            sequence_logprob(torch.full((2, 2), float('nan')), torch.tensor([0, 1]), 1)

    def test_opposite_questions_and_invalid_polarity(self):
        self.assertIn('同一个人', make_prompt('text', False))
        self.assertIn('不同的人', make_prompt('text', True))
        with self.assertRaises(ValueError):
            make_prompt('text', 1)


if __name__ == '__main__':
    unittest.main()
