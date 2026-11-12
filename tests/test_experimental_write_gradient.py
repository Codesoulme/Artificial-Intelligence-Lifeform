import unittest

import torch
import torch.nn.functional as F

from spora.experimental_write_gradient import controlled_write_target_gradient
from spora.integrated_open_base import NativeSPORAMemoryReader
from spora.temporal import TemporalConfig, TemporalSite


class ExperimentalWriteGradientTests(unittest.TestCase):
    def test_forward_and_target_gradient_match_explicit_differentiable_write(self):
        torch.manual_seed(321)
        cfg = TemporalConfig(width=8, heads=1, memory_width=6, expert_rank=2)
        sites = [TemporalSite(cfg), TemporalSite(cfg)]
        reader = NativeSPORAMemoryReader(8, 6, 0.5)
        # Nonzero readout so that target gradient is observable.
        with torch.no_grad():
            for site in reader.native_sites:
                site.read_out.weight.normal_()
        hidden = torch.randn(1, 3, 8)
        initial = [torch.randn(1, 6, 6) for _ in sites]
        writes = [F.normalize(torch.randn(1, 6), dim=-1) for _ in sites]
        reads = [torch.randn(1, 6) for _ in sites]
        gates = [torch.tensor([0.4]), torch.tensor([0.8])]
        target_a = torch.randn(1, 6, requires_grad=True)
        target_b = target_a.detach().clone().requires_grad_()
        actual, reference = [], []
        with controlled_write_target_gradient(reader, sites, target_a, writes, reads, gates):
            for i, site in enumerate(sites):
                mode = torch.zeros(1, dtype=torch.long)
                matrix, _ = site.write_temporal([initial[i], torch.zeros_like(initial[i])],
                    [torch.zeros_like(initial[i]), torch.zeros_like(initial[i])],
                    writes[i], target_a, gates[i], mode, False)
                actual.append(reader(hidden, [matrix[0]], i, reads[i]))
        for i, site in enumerate(sites):
            prior = torch.bmm(initial[i], writes[i].unsqueeze(-1)).squeeze(-1)
            update = (gates[i][:, None, None] * torch.sigmoid(site.log_lr)
                      * (target_b - prior).unsqueeze(-1) * writes[i].unsqueeze(1))
            # Reader detaches matrices by design: compute the same native read explicitly.
            matrix = initial[i] + update
            native = reader.native_sites[i]
            batch, length, width = hidden.shape
            query = F.normalize(reads[i], dim=-1).unsqueeze(1).expand(-1, length, -1).reshape(-1, 6)
            mat = matrix[:, None].expand(-1, length, -1, -1).reshape(-1, 6, 6)
            output = native.read(hidden.reshape(-1, 1, width), query, [mat]).reshape_as(hidden)
            reference.append(hidden + reader.strength * (output - hidden))
        for a, b in zip(actual, reference):
            self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-6))
        weights = [torch.randn_like(x) for x in actual]
        grad_a, = torch.autograd.grad(sum((x*w).sum() for x, w in zip(actual, weights)), target_a)
        grad_b, = torch.autograd.grad(sum((x*w).sum() for x, w in zip(reference, weights)), target_b)
        self.assertGreater(float(grad_b.norm()), 0)
        self.assertTrue(torch.allclose(grad_a, grad_b, atol=2e-6, rtol=2e-6))
        self.assertTrue(all(not s.read_out._forward_pre_hooks for s in reader.native_sites))

    def test_hooks_removed_if_body_raises(self):
        cfg = TemporalConfig(width=8, heads=1, memory_width=6, expert_rank=2)
        sites = [TemporalSite(cfg), TemporalSite(cfg)]
        reader = NativeSPORAMemoryReader(8, 6, 0.5)
        target = torch.ones(1, 6, requires_grad=True)
        addresses = [torch.ones(1, 6), torch.ones(1, 6)]
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with controlled_write_target_gradient(reader, sites, target, addresses, addresses,
                                                  [torch.ones(1), torch.ones(1)]):
                raise RuntimeError('sentinel')
        self.assertTrue(all(not s.read_out._forward_pre_hooks for s in reader.native_sites))


if __name__ == '__main__':
    unittest.main()
