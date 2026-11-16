import unittest

import torch

from spora.research_counterfactual_span_loss import counterfactual_span_loss, span_mean_log_probability


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
