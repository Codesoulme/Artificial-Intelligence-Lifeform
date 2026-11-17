import unittest
import torch
from spora.research_null_memory_residual import NullMemoryResidual


class NullResidualTests(unittest.TestCase):
    def test_initially_identity_and_receives_gradient(self):
        module = NullMemoryResidual(3)
        value = torch.ones(1, 2, 5)
        self.assertTrue(torch.equal(module(value), value))
        module(value).sum().backward()
        self.assertTrue(torch.equal(module.delta.grad, torch.tensor([2., 2., 2.])))

    def test_rejects_incompatible_input(self):
        with self.assertRaises(ValueError):
            NullMemoryResidual(3)(torch.ones(1, 2, 2))
