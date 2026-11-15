import unittest

import torch

from scripts import deepseek_spora_reasoning_position_distill as audit


class ReasoningPositionDistillTests(unittest.TestCase):
    def test_prefixes_exclude_prompt_and_unavailable_targets(self):
        self.assertEqual(audit.valid_lengths(289, 120), [160, 224, 288])
        self.assertEqual(audit.valid_lengths(224, 100), [160])
        with self.assertRaises(ValueError):
            audit.valid_lengths(20, 21)

    def test_teacher_agreement_and_sampled_token_loss(self):
        logits = torch.tensor([[1., 3., 0.]])
        result = audit.metrics(logits, logits.clone(), 1)
        self.assertTrue(result['teacher_top1_matches'])
        self.assertAlmostEqual(result['kl'], 0., places=6)
        self.assertGreater(result['sampled_token_nll'], 0.)


if __name__ == '__main__':
    unittest.main()
