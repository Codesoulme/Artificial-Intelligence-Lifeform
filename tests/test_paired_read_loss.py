import unittest

import torch

from spora.paired_read_loss import paired_read_loss


class PairedReadLossTests(unittest.TestCase):
    def test_improvement_lowers_loss(self):
        self.assertLess(paired_read_loss(torch.tensor([2.]), torch.tensor([-2.])),
                        paired_read_loss(torch.tensor([0.]), torch.tensor([0.])))

    def test_separate_backward_matches_joint_gradient(self):
        first = torch.tensor([.3, -.7], requires_grad=True)
        second = torch.tensor([-.4, .1], requires_grad=True)
        paired_read_loss(first, second).backward()
        expected = (first.grad.clone(), second.grad.clone())
        first.grad = None
        second.grad = None
        paired_read_loss(first, second.detach()).backward()
        paired_read_loss(first.detach(), second).backward()
        self.assertTrue(torch.allclose(first.grad, expected[0]))
        self.assertTrue(torch.allclose(second.grad, expected[1]))


if __name__ == '__main__':
    unittest.main()
