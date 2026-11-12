import contextvars
import unittest

import torch

from scripts.deepseek_spora_memory_gated_lowrank_pilot import MemoryGatedLowRankLinear


class MemoryGatedLowRankTests(unittest.TestCase):
    def test_no_memory_and_masked_read_preserve_base(self):
        torch.manual_seed(17)
        context = contextvars.ContextVar('test_memory', default=None)
        base = torch.nn.Linear(7, 5)
        layer = MemoryGatedLowRankLinear(base, context, rank=3)
        with torch.no_grad():
            layer.b.normal_()
        x = torch.randn(1, 2, 7)
        expected = base(x)
        self.assertTrue(torch.equal(layer(x), expected))
        token = context.set((None, None, torch.tensor([False]), None))
        try:
            self.assertTrue(torch.equal(layer(x), expected))
        finally:
            context.reset(token)
        token = context.set((None, None, torch.tensor([True]), None))
        try:
            self.assertFalse(torch.equal(layer(x), expected))
            with self.assertRaises(ValueError):
                layer(torch.randn(2, 2, 7))
        finally:
            context.reset(token)
        self.assertTrue(torch.equal(layer(x), expected))


if __name__ == '__main__':
    unittest.main()
