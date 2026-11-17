import unittest

import torch

from spora.research_counterfactual_span_loss import (
    compare_loss_gradients, counterfactual_span_loss, diagnostic_gradient_norms, span_mean_log_probability)


class CounterfactualSpanLossTests(unittest.TestCase):
    def test_correct_memory_value_alignment_reduces_loss(self):
        a = torch.tensor([[0, 1]])
        b = torch.tensor([[1, 0, 1]])
        a_good = torch.tensor([[[5., 0.], [0., 5.]]], requires_grad=True)
        b_good = torch.tensor([[[0., 5.], [5., 0.], [0., 5.]]], requires_grad=True)
        a_bad = -a_good.detach()
        b_bad = -b_good.detach()
        proper = counterfactual_span_loss(a_good, b_bad, a_bad, b_good, a, b)
        inverted = counterfactual_span_loss(a_bad, b_good, a_good, b_bad, a, b)
        self.assertLess(proper.item(), inverted.item())
        proper.backward()
        self.assertTrue(bool((a_good.grad.abs().sum(dim=-1) > 0).all()))
        self.assertTrue(bool((b_good.grad.abs().sum(dim=-1) > 0).all()))

    def test_padding_is_ignored_and_invalid_spans_are_rejected(self):
        scores = torch.tensor([[[2., 1.], [1., 2.], [99., -99.]]])
        target = torch.tensor([[0, 1, -100]])
        full = span_mean_log_probability(scores, target)
        short = span_mean_log_probability(scores[:, :2], target[:, :2])
        torch.testing.assert_close(full, short)
        with self.assertRaises(ValueError):
            span_mean_log_probability(scores, torch.full((1, 3), -100))
        with self.assertRaises(ValueError):
            span_mean_log_probability(scores, torch.tensor([[0, 1, 2]]))

    def test_diagnostic_finds_disconnected_and_does_not_write_gradients(self):
        reader = torch.nn.Parameter(torch.tensor(2.))
        unused = torch.nn.Parameter(torch.tensor(3.))
        adapter = torch.nn.Parameter(torch.tensor(4.))
        loss = (reader * adapter).square()
        result = diagnostic_gradient_norms(loss, {'reader': [reader, unused], 'adapter': [adapter]})
        self.assertEqual(result['reader']['connected'], 1)
        self.assertEqual(result['reader']['nonzero'], 1)
        self.assertGreater(result['reader']['norm'], 0)
        self.assertGreater(result['adapter']['norm'], 0)
        self.assertIsNone(reader.grad)
        loss.backward()
        self.assertIsNotNone(reader.grad)
        with self.assertRaises(ValueError):
            diagnostic_gradient_norms(loss, {'a': [reader], 'b': [reader]})

    def test_gradient_comparison_distinguishes_opposed_and_aligned_signals(self):
        p = torch.nn.Parameter(torch.tensor([1., 2.]))
        q = torch.nn.Parameter(torch.tensor([1.]))
        first = p.sum() + q.sum()
        other = -2 * p.sum() + 2 * q.sum()
        result = compare_loss_gradients(first, other, {'reader': [p], 'adapter': [q]})
        self.assertAlmostEqual(result['groups']['reader']['cosine'], -1.)
        self.assertAlmostEqual(result['groups']['adapter']['cosine'], 1.)
        self.assertAlmostEqual(result['total_norm'], 11 ** 0.5)
        self.assertLess(result['clip_factor'], 1.)
        self.assertIsNone(p.grad)
        (first + other).backward()
        torch.testing.assert_close(p.grad, torch.tensor([-1., -1.]))
