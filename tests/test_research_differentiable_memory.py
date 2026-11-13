import unittest

import torch
from torch import nn

from scripts.deepseek_spora_semantic_bridge_pilot import MemoryResidualOutput
from spora.integrated_open_base import IntegratedOpenBase
from spora.research_differentiable_memory import DifferentiablePacketWriter, differentiable_memory_context
from tests.test_integrated_open_base import Base, make_reader


class HeadBase(Base):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 3, bias=False)

    def forward(self, x, **kwargs):
        return self.head(super().forward(x, **kwargs))


class IdentityBridge(nn.Module):
    slot_count = 1

    def forward(self, x):
        return x


class DifferentiableMemoryTests(unittest.TestCase):
    def test_write_receives_gradient_and_preserves_other_orthogonal_address(self):
        writer = DifferentiablePacketWriter(3, 3)
        matrix = torch.zeros(1, 3, 3)
        first = torch.tensor([[1., 0., 0.]])
        second = torch.tensor([[0., 1., 0.]])
        a = writer(matrix, first, torch.tensor([[1., 2., 3.]]))
        b = writer(a, second, torch.tensor([[3., 2., 1.]]))
        torch.testing.assert_close(a.bmm(first.unsqueeze(-1)), b.bmm(first.unsqueeze(-1)))
        b.bmm(second.unsqueeze(-1)).sum().backward()
        self.assertGreater(float(writer.packet.weight.grad.norm()), 0.)
        self.assertTrue(torch.equal(matrix, torch.zeros_like(matrix)))

    def test_isolated_adapter_backpropagates_through_write(self):
        torch.manual_seed(42)
        decoder = IntegratedOpenBase(HeadBase(), make_reader(), (0, 2))
        adapter = MemoryResidualOutput(decoder.base.head, decoder._memory_context,
                                       3, IdentityBridge(), 3, 1)
        with torch.no_grad():
            adapter.to_hidden.weight.copy_(torch.eye(3))
        decoder.base.head = adapter
        decoder.requires_grad_(False)
        writer = DifferentiablePacketWriter(3, 3)
        address = torch.tensor([[1., 0., 0.]])
        observation = torch.tensor([[1., 2., 3.]])
        matrices = [writer(torch.zeros(1, 3, 3), address, observation) for _ in range(2)]
        x = torch.tensor([[[0.2, -0.5, 0.1]]])
        plain = decoder(x)
        with differentiable_memory_context(decoder, matrices, (address, address)):
            output = decoder.base(x)
            with self.assertRaisesRegex(ValueError, 'Nested'):
                with differentiable_memory_context(decoder, matrices, (address, address)):
                    pass
        self.assertFalse(torch.equal(output, plain))
        output.square().sum().backward()
        self.assertGreater(float(writer.packet.weight.grad.norm()), 0.)
        self.assertTrue(all(param.grad is None for param in decoder.parameters()))
        self.assertTrue(torch.equal(decoder(x), plain))
        self.assertIsNone(decoder._memory_context.get())

    def test_context_is_reset_after_exception_and_bad_address_is_rejected(self):
        decoder = IntegratedOpenBase(HeadBase(), make_reader(), (0, 2))
        matrix = torch.zeros(1, 3, 3)
        address = torch.tensor([[1., 0., 0.]])
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with differentiable_memory_context(decoder, (matrix, matrix), (address, address)):
                raise RuntimeError('injected')
        self.assertIsNone(decoder._memory_context.get())
        with self.assertRaises(ValueError):
            with differentiable_memory_context(decoder, (matrix, matrix), (address, torch.zeros_like(address))):
                pass

    def test_explicit_inactive_slot_does_not_receive_writer_gradient(self):
        writer = DifferentiablePacketWriter(3, 3)
        address = torch.tensor([[1., 0., 0.]])
        mask = torch.tensor([[True, True, False]])
        matrix = writer(torch.zeros(1, 3, 3), address, torch.ones(1, 3), active_mask=mask)
        matrix.sum().backward()
        self.assertEqual(float(writer.packet.weight.grad[2].abs().max()), 0.)
        self.assertGreater(float(writer.packet.weight.grad[:2].norm()), 0.)


if __name__ == '__main__':
    unittest.main()
