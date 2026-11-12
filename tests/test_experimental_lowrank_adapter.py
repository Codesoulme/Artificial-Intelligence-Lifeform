import unittest

import torch

from scripts.deepseek_spora_joint_lowrank_pilot import LowRankLinear


class LowRankAdapterTests(unittest.TestCase):
    def test_zero_initialization_freezes_base_and_disable_restores_output(self):
        torch.manual_seed(41)
        base = torch.nn.Linear(7, 5)
        inputs = torch.randn(2, 3, 7)
        expected = base(inputs).detach().clone()
        adapter = LowRankLinear(base, rank=3)
        self.assertTrue(torch.equal(adapter(inputs), expected))
        self.assertFalse(any(p.requires_grad for p in adapter.base.parameters()))
        with torch.no_grad():
            adapter.b.normal_()
        self.assertFalse(torch.equal(adapter(inputs), expected))
        adapter.enabled = False
        self.assertTrue(torch.equal(adapter(inputs), expected))
        adapter.enabled = True
        adapter(inputs).square().sum().backward()
        self.assertIsNotNone(adapter.a.grad)
        self.assertIsNotNone(adapter.b.grad)
        self.assertTrue(all(p.grad is None for p in adapter.base.parameters()))


if __name__ == '__main__':
    unittest.main()
