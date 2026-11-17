import unittest
import torch
from spora.research_first_token_pair_loss import pair_loss


class PairLossTests(unittest.TestCase):
    def test_pair_direction_and_gradient(self):
        a=torch.tensor([[2.,0.]],requires_grad=True)
        b=torch.tensor([[0.,2.]],requires_grad=True)
        self.assertLess(pair_loss(a,b,0,1),pair_loss(b,a,0,1))
        pair_loss(a,b,0,1).backward()
        self.assertLess(a.grad[0,0],0)
        self.assertGreater(a.grad[0,1],0)
        self.assertLess(b.grad[0,1],0)

    def test_same_label_is_invalid(self):
        with self.assertRaises(ValueError):pair_loss(torch.zeros(1,2),torch.zeros(1,2),0,0)
