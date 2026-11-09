from types import SimpleNamespace

import unittest
import torch
from torch import nn

from spora.open_base_memory import OpenBaseMemory, memory_hooks
from spora.sleep_memory import SealedPage

check = unittest.TestCase()


def state(width=3, batch=1):
    current = [torch.eye(width).repeat(batch, 1, 1).requires_grad_() for _ in range(2)]
    page = [[(2 * torch.eye(width)).unsqueeze(0).requires_grad_(), torch.eye(width).unsqueeze(0)] for _ in range(2)]
    return SimpleNamespace(current=current, pages=[page])


def configured():
    adapter = OpenBaseMemory(3, 3)
    with torch.no_grad():
        for q, out in zip(adapter.query, adapter.out):
            q.weight.copy_(torch.eye(3))
            out.weight.copy_(torch.eye(3))
        adapter.gate.zero_()
    return adapter


def test_zero_initialization_and_injection_with_half_hidden():
    layers = nn.ModuleList([nn.Identity(), nn.Identity()])
    adapter = OpenBaseMemory(3, 3)
    x = torch.randn(2, 4, 3, dtype=torch.float16)
    with memory_hooks(layers, adapter, state()):
        assert torch.equal(layers[1](layers[0](x)), x)
    adapter = configured()
    with memory_hooks(layers, adapter, state()):
        result = layers[1](layers[0](x))
        assert result.dtype == x.dtype
        assert not torch.equal(result, x)
    assert torch.equal(layers[1](layers[0](x)), x)


def test_read_formula_and_snapshot_lifecycle():
    adapter = configured()
    layers = [nn.Identity(), nn.Identity()]
    memory = state()
    x = torch.tensor([[[1., 0., 0.]]])
    expected = x + .5 * (1 + torch.softmax(torch.tensor([1., 2.]), 0)[1]) * x
    with memory_hooks(layers, adapter, memory):
        assert torch.allclose(layers[0](x), expected)
        with torch.no_grad():
            memory.current[0].zero_()
        assert torch.allclose(layers[0](x), expected)
    with memory_hooks(layers, adapter, memory):
        assert not torch.allclose(layers[0](x), expected)


def test_gradients_only_adapter_when_base_frozen():
    layers = nn.ModuleList([nn.Linear(3, 3), nn.Linear(3, 3)])
    layers.requires_grad_(False)
    adapter = configured()
    memory = state()
    with memory_hooks(layers, adapter, memory):
        layers[1](layers[0](torch.randn(2, 4, 3))).square().sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())
    assert all(p.grad is None for p in layers.parameters())
    assert all(t.grad is None for t in memory.current)
    assert all(site[0].grad is None for site in memory.pages[0])


def test_sealed_page_and_batch_specific_matrices():
    memory = state(batch=2)
    factors = [[torch.eye(3).requires_grad_(), torch.eye(3).requires_grad_(), torch.ones(6)] for _ in range(2)]
    memory.pages = [SealedPage(factors, 3)]
    layers = [nn.Identity(), nn.Identity()]
    with memory_hooks(layers, configured(), memory):
        assert layers[0](torch.ones(2, 1, 3)).shape == (2, 1, 3)
        with check.assertRaisesRegex(ValueError, 'Memory matrices'):
            layers[0](torch.ones(3, 1, 3))
    assert all(factor.grad is None for site in factors for factor in site)


def test_exception_removes_hooks():
    layers = [nn.Identity(), nn.Identity()]
    with check.assertRaisesRegex(RuntimeError, 'failure'):
        with memory_hooks(layers, configured(), state()):
            raise RuntimeError('failure')
    assert not layers[0]._forward_hooks and not layers[1]._forward_hooks
    assert torch.equal(layers[0](torch.ones(1, 1, 3)), torch.ones(1, 1, 3))


def test_invalid_shapes_and_layer_outputs():
    adapter = configured()
    with check.assertRaises(ValueError):
        OpenBaseMemory(0, 3)
    with check.assertRaises(ValueError):
        with memory_hooks([nn.Identity()], adapter, state()):
            pass
    with check.assertRaisesRegex(ValueError, 'Layer output'):
        adapter(torch.ones(2, 3), [torch.eye(3).unsqueeze(0)], 0)
    with check.assertRaisesRegex(ValueError, 'Memory matrices'):
        adapter(torch.ones(2, 1, 3), [torch.ones(1, 2, 3)], 0)
    class TupleLayer(nn.Module):
        def forward(self, x):
            return (x,)
    layer = TupleLayer()
    with check.assertRaisesRegex(ValueError, 'Layer output'):
        with memory_hooks([layer, nn.Identity()], adapter, state()):
            layer(torch.ones(1, 1, 3))
    assert not layer._forward_hooks


class OpenBaseMemoryTests(unittest.TestCase):
    def test_overlapping_contexts_fail_without_removing_original_hooks(self):
        layers = [nn.Identity(), nn.Identity()]
        adapter = configured()
        x = torch.ones(1, 1, 3)
        with memory_hooks(layers, adapter, state()):
            expected = layers[0](x)
            with self.assertRaisesRegex(ValueError, 'Overlapping'):
                with memory_hooks(layers, adapter, state()):
                    pass
            self.assertTrue(torch.equal(layers[0](x), expected))
        self.assertTrue(torch.equal(layers[0](x), x))

    def test_zero_initialization_and_injection_with_half_hidden(self):
        test_zero_initialization_and_injection_with_half_hidden()

    def test_read_formula_and_snapshot_lifecycle(self):
        test_read_formula_and_snapshot_lifecycle()

    def test_gradients_only_adapter_when_base_frozen(self):
        test_gradients_only_adapter_when_base_frozen()

    def test_sealed_page_and_batch_specific_matrices(self):
        test_sealed_page_and_batch_specific_matrices()

    def test_exception_removes_hooks(self):
        test_exception_removes_hooks()

    def test_invalid_shapes_and_layer_outputs(self):
        test_invalid_shapes_and_layer_outputs()
