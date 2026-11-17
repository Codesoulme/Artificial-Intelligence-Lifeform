import unittest
import torch
from spora.research_partitioned_attention import partitioned_attention


class PartitionTests(unittest.TestCase):
    def test_reference_matches_masked_attention_and_gradients(self):
        torch.manual_seed(21)
        q = torch.randn(2, 3, 4, requires_grad=True)
        k = torch.randn(2, 5, 4, requires_grad=True)
        v = torch.randn(2, 5, 7, requires_grad=True)
        history = torch.tensor([[[0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0]]] * 2, dtype=torch.bool)
        local = torch.tensor([[[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]]] * 2, dtype=torch.bool)
        out, mass, _, _ = partitioned_attention(q, k, v, history, local)
        scores = (q @ k.transpose(-1, -2) / 2).masked_fill(~(history | local), -torch.inf)
        reference = torch.softmax(scores, -1) @ v
        torch.testing.assert_close(out, reference)
        self.assertEqual(mass[0, 0].item(), 0)
        actual = torch.autograd.grad(out.square().sum(), (q, k, v), retain_graph=True)
        expected = torch.autograd.grad(reference.square().sum(), (q, k, v))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_rejects_overlap_and_no_visible_keys(self):
        q, k, v = torch.ones(1, 1, 2), torch.ones(1, 2, 2), torch.ones(1, 2, 2)
        full = torch.ones(1, 1, 2, dtype=torch.bool)
        empty = ~full
        with self.assertRaises(ValueError):
            partitioned_attention(q, k, v, full, full)
        with self.assertRaises(ValueError):
            partitioned_attention(q, k, v, empty, empty)
