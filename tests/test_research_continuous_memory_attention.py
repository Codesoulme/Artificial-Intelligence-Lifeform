import unittest
from types import SimpleNamespace
import torch
from torch import nn
from spora.research_continuous_memory_attention import ContinuousMemoryAttention


class MemoryContext:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value


def make_state(values, enabled=True):
    address = torch.zeros(1, 12)
    address[0, 0] = 1
    data = torch.zeros(12)
    data[:8] = torch.tensor(values, dtype=torch.float32)
    matrix = torch.outer(data, address[0]).unsqueeze(0)
    return ([[matrix], [matrix.clone()]], [address, address.clone()],
            torch.tensor([enabled]), torch.tensor([0]))


class ContinuousMemoryAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.context = MemoryContext()
        codec = SimpleNamespace(memory_width=12, slot_width=4, slot_count=3)
        self.layer = ContinuousMemoryAttention(nn.Linear(6, 6, bias=False), self.context,
                                               12, codec, None, slots=2, width=4)
        self.x = torch.randn(1, 3, 6)

    def test_zero_initialization_and_no_memory_preserve_base(self):
        expected = self.layer.base(self.x)
        torch.testing.assert_close(self.layer(self.x), expected)
        self.context.value = make_state([1, 2, 3, 4, 4, 3, 2, 1])
        torch.testing.assert_close(self.layer(self.x), expected)
        self.layer(self.x).sum().backward()
        self.assertGreater(self.layer.out.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(self.layer.base.weight.grad)

    def test_switching_memory_changes_output_and_mask_disables_it(self):
        with torch.no_grad():
            self.layer.out.weight.normal_()
        self.context.value = make_state([1, 2, 3, 4, 4, 3, 2, 1])
        first = self.layer(self.x)
        self.context.value = make_state([4, 2, 1, 3, -1, -2, 3, 1])
        second = self.layer(self.x)
        self.assertFalse(torch.allclose(first, second))
        self.context.value = make_state([4, 2, 1, 3, -1, -2, 3, 1], enabled=False)
        torch.testing.assert_close(self.layer(self.x), self.layer.base(self.x))
        self.context.value = make_state([0] * 8)
        torch.testing.assert_close(self.layer(self.x), self.layer.base(self.x))

    def test_snapshot_roundtrip_and_invalid_address(self):
        saved = self.layer.memory_state()
        with torch.no_grad():
            self.layer.query.weight.add_(1)
        self.layer.load_memory_state(saved)
        torch.testing.assert_close(self.layer.query.weight, saved['query']['weight'])
        state = list(make_state([1, 2, 3, 4, 4, 3, 2, 1]))
        state[1] = [torch.ones(1, 10), torch.ones(1, 10)]
        self.context.value = tuple(state)
        with self.assertRaises(ValueError):
            self.layer(self.x)

    def test_nonzero_output_layer_connects_attention_parameters(self):
        with torch.no_grad():
            self.layer.out.weight.normal_()
        self.context.value = make_state([1, 2, 3, 4, 4, 3, 2, 1])
        self.layer(self.x).square().sum().backward()
        for name in ('query', 'key', 'value', 'out'):
            gradient = getattr(self.layer, name).weight.grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)

    def test_output_gain_scales_only_active_memory_residual(self):
        codec = SimpleNamespace(memory_width=12, slot_width=4, slot_count=3)
        stronger = ContinuousMemoryAttention(nn.Linear(6, 6, bias=False), self.context,
                                            12, codec, None, slots=2, width=4, output_gain=10.)
        with torch.no_grad():
            self.layer.out.weight.normal_()
        stronger.load_state_dict(self.layer.state_dict(), strict=True)
        self.context.value = make_state([1, 2, 3, 4, 4, 3, 2, 1])
        base = self.layer.base(self.x)
        torch.testing.assert_close(stronger(self.x) - base, 10 * (self.layer(self.x) - base),
                                   atol=1e-5, rtol=1e-5)
        self.context.value = None
        torch.testing.assert_close(stronger(self.x), base)
        with self.assertRaises(ValueError):
            ContinuousMemoryAttention(nn.Linear(6, 6), self.context, 12, codec, None,
                                      slots=2, width=4, output_gain=float('nan'))
