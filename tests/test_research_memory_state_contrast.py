import unittest
import torch

from spora.research_memory_state_contrast import memory_state_contrast_loss


class StateContrastTests(unittest.TestCase):
    def test_rewards_matching_state_for_each_candidate(self):
        target_a = torch.tensor([[0, 1]])
        target_b = torch.tensor([[1, 0, 1]])
        a = torch.tensor([[[3., 0.], [0., 3.]]], requires_grad=True)
        b = torch.tensor([[[0., 3.], [3., 0.], [0., 3.]]], requires_grad=True)
        wrong_a, wrong_b = -a.detach(), -b.detach()
        good = memory_state_contrast_loss(a, wrong_b, wrong_a, b, target_a, target_b)
        bad = memory_state_contrast_loss(wrong_a, b, a, wrong_b, target_a, target_b)
        self.assertLess(good.item(), bad.item())
        good.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertTrue(torch.isfinite(b.grad).all())

    def test_invalid_margin_rejected(self):
        t = torch.tensor([[0]])
        x = torch.zeros((1, 1, 2))
        for margin in (-1, float('nan'), True):
            with self.assertRaises(ValueError):
                memory_state_contrast_loss(x, x, x, x, t, t, margin)
