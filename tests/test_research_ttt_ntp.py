import unittest

import torch
from torch import nn

from spora.research_ttt_ntp import (TTTNTPMemoryLinear, attach_read_adapters,
                                    fresh_memory, install, sequential_ntp_write,
                                    use_memory)


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            nn.Module() for _ in range(2)
        ])
        for layer in self.model.layers:
            layer.mlp = nn.Module()
            layer.mlp.down_proj = nn.Linear(3, 2, bias=False)


class ResearchTTTNTPTests(unittest.TestCase):
    def test_install_and_absent_memory_preserve_native_layer(self):
        decoder = ToyDecoder()
        wrappers = install(decoder, (0, 1))
        hidden = torch.tensor([[[1., 2., 3.]]])
        expected = wrappers[0].base(hidden)
        self.assertTrue(torch.equal(wrappers[0](hidden), expected))
        self.assertIsInstance(decoder.model.layers[0].mlp.down_proj, TTTNTPMemoryLinear)

    def test_causal_write_changes_only_active_call_and_matches_target(self):
        decoder = ToyDecoder()
        wrappers = install(decoder, (0, 1))
        memory = fresh_memory(wrappers)
        keys = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        targets = {0: torch.tensor([[2., 3.], [4., 5.]]),
                   1: torch.tensor([[2., 3.], [4., 5.]])}
        sequential_ntp_write(memory, keys, targets, eta=1., ridge=0.001)
        hidden = keys[:1].reshape(1, 1, 3)
        ordinary = wrappers[0](hidden)
        with use_memory(memory):
            adapted = wrappers[0](hidden)
        self.assertFalse(torch.equal(ordinary, adapted))
        torch.testing.assert_close(adapted[0, 0] - ordinary[0, 0], targets[0][0],
                                  atol=0.02, rtol=0.02)
        self.assertTrue(torch.equal(wrappers[0](hidden), ordinary))

    def test_read_adapter_is_zero_initialized_and_trainable(self):
        decoder = ToyDecoder()
        wrappers = install(decoder, (0,))
        adapters = attach_read_adapters(wrappers, rank=2)
        memory = fresh_memory(wrappers)
        keys = torch.tensor([[1., 0., 0.]])
        sequential_ntp_write(memory, keys, {0: torch.tensor([[2., 3.]])}, eta=1., ridge=0.001)
        hidden = keys.reshape(1, 1, 3)
        with use_memory(memory):
            before = wrappers[0](hidden)
        self.assertTrue(torch.equal(before, wrappers[0].base(hidden)))
        loss = before.square().sum()
        loss.backward()
        self.assertIsNotNone(adapters[0].up.weight.grad)

    def test_layer_keys_and_alignment(self):
        decoder = ToyDecoder()
        wrappers = install(decoder, (0, 1))
        memory = fresh_memory(wrappers)
        keys = {0: torch.tensor([[1., 0., 0.]]), 1: torch.tensor([[0., 1., 0.]])}
        targets = {0: torch.tensor([[2., 3.]]), 1: torch.tensor([[4., 5.]])}
        sequential_ntp_write(memory, keys, targets, eta=1., ridge=0.001)
        self.assertGreater(float(memory[0][:, 0].norm()), 0.)
        self.assertEqual(float(memory[0][:, 1].norm()), 0.)
        self.assertGreater(float(memory[1][:, 1].norm()), 0.)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            sequential_ntp_write({0: torch.zeros(2, 3)}, {0: torch.ones(2, 3)},
                                 {0: torch.ones(1, 2)})


if __name__ == '__main__':
    unittest.main()
