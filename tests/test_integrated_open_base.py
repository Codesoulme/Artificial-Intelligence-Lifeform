import unittest
from types import SimpleNamespace

import torch
from torch import nn

from spora.integrated_open_base import IntegratedOpenBase
from spora.open_base_memory import OpenBaseMemory, memory_hooks


class Decoder(nn.Module):
    attention_type = 'full_attention'

    def forward(self, x):
        return x + 0.1


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Decoder(), Decoder(), Decoder()])

    def forward(self, x, fail=False, **kwargs):
        for layer in self.model.layers:
            x = layer(x)
        if fail:
            raise RuntimeError('failure')
        return x

    def generate(self, x, **kwargs):
        return self.forward(x, **kwargs)


def make_state(scale):
    return SimpleNamespace(current=[torch.eye(3).unsqueeze(0) * scale for _ in range(2)], pages=[])


def make_reader():
    reader = OpenBaseMemory(3, 3)
    with torch.no_grad():
        for query, output in zip(reader.query, reader.out):
            query.weight.copy_(torch.eye(3))
            output.weight.copy_(torch.eye(3))
        reader.gate.zero_()
    return reader


class IntegratedOpenBaseTests(unittest.TestCase):
    def test_matches_existing_hooks_for_multiple_states_and_generate(self):
        base, reader = Base(), make_reader()
        x = torch.tensor([[[1., 0., 0.]]])
        memories = [make_state(1.), make_state(2.)]
        addresses = [torch.tensor([[1., 0., 0.]]), torch.tensor([[1., 0., 0.]])]
        plain = base(x)
        expected = []
        for memory in memories:
            with memory_hooks([base.model.layers[0], base.model.layers[2]], reader, memory,
                              current_only=True, addresses=addresses):
                expected.append(base(x))
        integrated = IntegratedOpenBase(base, reader, (0, 2))
        self.assertTrue(torch.equal(integrated(x), plain))
        for memory, target in zip(memories, expected):
            self.assertTrue(torch.equal(integrated(x, memory_state=memory, memory_addresses=addresses,
                                                   current_only=True), target))
            self.assertTrue(torch.equal(integrated.generate(x, memory_state=memory, memory_addresses=addresses,
                                                            current_only=True), target))
        self.assertFalse(torch.equal(expected[0], expected[1]))
        self.assertFalse(any(layer._forward_hooks for layer in base.model.layers))

    def test_error_cleans_state_and_external_cache_is_rejected(self):
        base = Base()
        model = IntegratedOpenBase(base, make_reader(), (0, 2))
        x = torch.ones(1, 1, 3)
        plain = model(x)
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            model(x, memory_state=make_state(1), fail=True)
        self.assertTrue(torch.equal(model(x), plain))
        with self.assertRaisesRegex(ValueError, 'caches'):
            model(x, memory_state=make_state(1), past_key_values=object())
        with self.assertRaisesRegex(ValueError, 'positional'):
            model(x, None, memory_state=make_state(1))
        with self.assertRaises(ValueError):
            model(x, memory_addresses=[torch.ones(1, 3)] * 2)

    def test_adapter_receives_gradients_and_memory_is_detached(self):
        reader = make_reader()
        model = IntegratedOpenBase(Base(), reader, (0, 2))
        memory = make_state(1.)
        for matrix in memory.current:
            matrix.requires_grad_()
        model(torch.ones(1, 1, 3), memory_state=memory, current_only=True).sum().backward()
        self.assertTrue(all(p.grad is not None for p in reader.parameters()))
        self.assertTrue(all(t.grad is None for t in memory.current))


if __name__ == '__main__':
    unittest.main()
