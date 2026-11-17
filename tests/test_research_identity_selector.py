import unittest
import torch
from spora.research_identity_selector import IdentitySelector


class IdentitySelectorTests(unittest.TestCase):
    def test_candidate_permutation_and_null(self):
        torch.manual_seed(2)
        model=IdentitySelector(width=8,rank=4)
        q=torch.randn(3,8);r=torch.randn(3,4,8)
        a=model(q,r);b=model(q,r[:,[2,0,3,1]])
        torch.testing.assert_close(b[:,:4],a[:,[2,0,3,1]])
        torch.testing.assert_close(a[:,-1],b[:,-1])
        torch.nn.functional.cross_entropy(a,torch.tensor([1,4,2])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_shape_validation(self):
        with self.assertRaises(ValueError):IdentitySelector(8,4)(torch.zeros(1,8),torch.zeros(1,2,7))
