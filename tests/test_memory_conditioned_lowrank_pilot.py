import contextvars
import unittest

import torch

from scripts.deepseek_spora_memory_conditioned_lowrank_pilot import MemoryGatedLowRankLinear


class MemoryConditionedLowRankTests(unittest.TestCase):
    def test_zero_memory_is_neutral_and_values_change_result(self):
        torch.manual_seed(19)
        context = contextvars.ContextVar('test_memory', default=None)
        base = torch.nn.Linear(7, 5)
        layer = MemoryGatedLowRankLinear(base, context, memory_width=6, rank=3)
        with torch.no_grad():
            layer.b.normal_()
        x = torch.randn(1, 2, 7)
        expected = base(x)
        self.assertTrue(torch.equal(layer(x), expected))
        zero = torch.zeros(1, 6, 6)
        addresses = (torch.eye(6)[0:1], torch.eye(6)[0:1])
        slots = torch.tensor([0])
        def run(matrix, mask=None):
            token = context.set((((matrix,), (matrix,)), addresses, mask, slots))
            try:
                return layer(x)
            finally:
                context.reset(token)
        self.assertTrue(torch.equal(run(zero), expected))
        matrix = torch.randn(1, 6, 6)
        self.assertTrue(torch.equal(run(matrix, torch.tensor([False])), expected))
        value_a, value_b = run(matrix), run(-matrix)
        self.assertFalse(torch.equal(value_a, expected))
        self.assertFalse(torch.equal(value_a, value_b))
        value_a.square().sum().backward()
        self.assertIsNotNone(layer.condition.grad)
        self.assertIsNotNone(layer.a.grad)
        self.assertIsNotNone(layer.b.grad)
        self.assertTrue(all(param.grad is None for param in base.parameters()))


if __name__ == '__main__':
    unittest.main()
