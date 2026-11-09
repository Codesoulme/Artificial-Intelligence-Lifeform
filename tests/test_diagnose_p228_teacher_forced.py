import math
import unittest

import torch

from scripts.diagnose_p228_teacher_forced import score_logits


class FixedHistoryScoreTests(unittest.TestCase):
    def test_masks_forbidden_and_scores_termination(self):
        logits = torch.tensor([[100., 3., 1., 0.], [100., 0., 1., 4.]])
        result = score_logits(logits, torch.tensor([1, 3]), eos=3, forbidden=(0,), ended=True)
        self.assertEqual(result['targets'], 2)
        self.assertEqual(result['target_greedy_matches'], 2)
        self.assertTrue(result['eos_greedy'])
        self.assertEqual(result['eos_rank'], 1)
        expected = math.log(math.exp(0)+math.exp(1)+math.exp(4))-4
        self.assertAlmostEqual(result['eos_nll'], expected, places=6)

    def test_checks_shapes_and_terminal_identity(self):
        scores = torch.zeros(2, 4)
        with self.assertRaises(ValueError):
            score_logits(scores, torch.tensor([1]), eos=3, forbidden=(0,), ended=False)
        with self.assertRaises(ValueError):
            score_logits(scores, torch.tensor([1, 2]), eos=3, forbidden=(0,), ended=True)
        with self.assertRaises(ValueError):
            score_logits(scores, torch.tensor([1, 3]), eos=3, forbidden=(3,), ended=True)


if __name__ == '__main__':
    unittest.main()
