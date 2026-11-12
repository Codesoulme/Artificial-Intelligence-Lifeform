import unittest
from contextvars import ContextVar

from scripts import deepseek_spora_semantic_bridge_pilot as s

torch = s.p.torch


class MemoryResidualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.context = ContextVar('test_memory', default=None)
        self.bridge = s.TokenCodeDecoder(2, 4)
        self.bridge.slot_count = 5
        self.base = torch.nn.Linear(4, 4)
        self.layer = s.MemoryResidualOutput(self.base, self.context, 10, self.bridge, 4, 4)
        with torch.no_grad():
            self.layer.to_hidden.weight.normal_()
        self.x = torch.randn(1, 3, 4)
        self.address = torch.eye(10)[:1]

    def call_with(self, matrix, mask=True):
        active = (((matrix,), (matrix,)), (self.address, self.address),
                  torch.tensor([mask]), torch.tensor([0]))
        token = self.context.set(active)
        try:
            return self.layer(self.x)
        finally:
            self.context.reset(token)

    def test_off_zero_and_masked_paths_preserve_base(self):
        expected = self.base(self.x)
        self.assertTrue(torch.equal(self.layer(self.x), expected))
        self.assertTrue(torch.equal(self.call_with(torch.zeros(1, 10, 10)), expected))
        self.assertTrue(torch.equal(self.call_with(torch.randn(1, 10, 10), False), expected))

    def test_nonzero_memory_has_trainable_residual_without_state_mutation(self):
        matrix = torch.randn(1, 10, 10)
        saved = matrix.clone()
        output = self.call_with(matrix)
        self.assertFalse(torch.equal(output, self.base(self.x)))
        output.square().sum().backward()
        self.assertGreater(float(self.layer.to_hidden.weight.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in self.base.parameters()))
        self.assertTrue(all(p.grad is None for p in self.bridge.parameters()))
        self.assertTrue(torch.equal(matrix, saved))


if __name__ == '__main__':
    unittest.main()
