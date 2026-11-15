import unittest

import torch
from torch import nn
from spora.research_selective_reader import SelectiveMemoryReader


class FixedReader(nn.Module):
    def forward(self, hidden, matrices, site, address=None):
        return hidden + 2.0


class SelectiveReaderTests(unittest.TestCase):
    def test_zero_and_one_gate_preserve_baseline_and_apply_reader(self):
        module = SelectiveMemoryReader(FixedReader(), 3)
        x = torch.randn(1, 4, 3)
        with torch.no_grad():
            for gate in module.gates:
                gate.weight.zero_()
                gate.bias.fill_(-10.)
        self.assertTrue(torch.equal(module(x, [], 0), x))
        with torch.no_grad():
            module.gates[0].bias.fill_(10.)
        torch.testing.assert_close(module(x, [], 0), x + 2.)
        self.assertTrue(torch.equal(module(x, [], 1), x))
        module.enabled = False
        torch.testing.assert_close(module(x, [], 1), x + 2.)

    def test_gate_is_token_local(self):
        module = SelectiveMemoryReader(FixedReader(), 3)
        x = torch.randn(1, 5, 3)
        changed = x.clone()
        changed[:, 3:] += torch.randn_like(changed[:, 3:]) * 5
        torch.testing.assert_close(module(x, [], 0)[:, :3], module(changed, [], 0)[:, :3])


if __name__ == '__main__':
    unittest.main()
