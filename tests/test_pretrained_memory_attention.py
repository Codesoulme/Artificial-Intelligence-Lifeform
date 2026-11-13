import unittest
from contextvars import ContextVar
from types import SimpleNamespace

import torch
from torch import nn

from spora.research_pretrained_memory_attention import PretrainedMemoryAttention, disable_memory_attention


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 2
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 2, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(4))
            self.k_proj.weight.copy_(torch.eye(4)[:2])
            self.v_proj.weight.copy_(torch.eye(4)[:2])
            self.o_proj.weight.copy_(torch.eye(4))

    def forward(self, hidden_states, **kwargs):
        return hidden_states + 0.1, None


class PretrainedMemoryAttentionTests(unittest.TestCase):
    def setUp(self):
        self.context = ContextVar('memory', default=None)
        self.codec = SimpleNamespace(slot_count=2, slot_width=2, codebook=torch.eye(2))
        self.embedding = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        self.module = PretrainedMemoryAttention(ToyAttention(), nn.Identity(),
                         SimpleNamespace(_memory_context=self.context), self.codec, self.embedding, slots=2)
        self.hidden = torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.]]])
        address = torch.tensor([[1., 0., 0., 0.]])
        matrix = torch.zeros(1, 4, 4)
        matrix[0, :, 0] = torch.tensor([1., 0., 0., 1.])
        self.active = (((matrix,), (matrix,)), (address, address), None, None)

    def test_zero_initialization_and_absent_memory_preserve_output(self):
        expected = self.module.inner(hidden_states=self.hidden)[0]
        self.assertTrue(torch.equal(self.module(hidden_states=self.hidden)[0], expected))
        token = self.context.set(self.active)
        try:
            actual = self.module(hidden_states=self.hidden)[0]
            self.assertTrue(torch.equal(actual, expected))
            actual.square().sum().backward()
            self.assertIsNotNone(self.module.gain.grad)
            self.assertGreater(abs(float(self.module.gain.grad)), 0.)
        finally:
            self.context.reset(token)

    def test_nonzero_gain_changes_output_and_gradients_reach_memory_projections(self):
        with torch.no_grad():
            self.module.gain.fill_(1.)
        token = self.context.set(self.active)
        try:
            actual = self.module(hidden_states=self.hidden)[0]
            expected = self.module.inner(hidden_states=self.hidden)[0]
            self.assertFalse(torch.equal(actual, expected))
            actual.square().sum().backward()
            self.assertGreater(float(self.module.memory_key.weight.grad.norm()), 0.)
            self.assertGreater(float(self.module.memory_value.weight.grad.norm()), 0.)
        finally:
            self.context.reset(token)

    def test_empty_slots_skip_residual(self):
        with torch.no_grad():
            self.module.gain.fill_(1.)
        matrix = torch.zeros(1, 4, 4)
        address = torch.tensor([[1., 0., 0., 0.]])
        token = self.context.set((((matrix,), (matrix,)), (address, address), None, None))
        try:
            self.assertTrue(torch.equal(self.module(hidden_states=self.hidden)[0],
                                        self.module.inner(hidden_states=self.hidden)[0]))
        finally:
            self.context.reset(token)

    def test_switching_memory_changes_output_with_fixed_input(self):
        with torch.no_grad():
            self.module.gain.fill_(1.)
        address = torch.tensor([[1., 0., 0., 0.]])
        outputs = []
        for packet in ([1., 0., 0., 0.], [0., 1., 0., 0.]):
            matrix = torch.zeros(1, 4, 4)
            matrix[0, :, 0] = torch.tensor(packet)
            token = self.context.set((((matrix,), (matrix,)), (address, address), None, None))
            try:
                outputs.append(self.module(hidden_states=self.hidden)[0])
            finally:
                self.context.reset(token)
        self.assertFalse(torch.equal(outputs[0], outputs[1]))

    def test_disable_restores_gain_after_error(self):
        with torch.no_grad():
            self.module.gain.fill_(0.75)
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with disable_memory_attention(self.module):
                self.assertEqual(float(self.module.gain), 0.)
                raise RuntimeError('injected')
        self.assertEqual(float(self.module.gain), 0.75)


if __name__ == '__main__':
    unittest.main()
