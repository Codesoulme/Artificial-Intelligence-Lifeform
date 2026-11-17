import unittest
import torch
from torch import nn
from spora.research_question_conditioned_span import QuestionConditionedSpan


class SimpleBase(nn.Module):
    def forward(self, x, mask):
        return torch.zeros(x.shape[0], 1 + x.shape[1] ** 2)


class QuestionConditionedSpanTests(unittest.TestCase):
    def test_initial_equivalence_and_question_sensitivity(self):
        torch.manual_seed(1)
        base = SimpleBase()
        head = QuestionConditionedSpan(base, 4, rank=2)
        x = torch.randn(1, 3, 4)
        mask = torch.ones(1, 3, dtype=torch.bool)
        q1 = torch.tensor([[False, False, True]])
        q2 = torch.tensor([[False, True, False]])
        self.assertTrue(torch.equal(head(x, mask, q1), base(x, mask)))
        with torch.no_grad():
            head.start.weight.fill_(.2)
        self.assertFalse(torch.allclose(head(x, mask, q1), head(x, mask, q2)))

    def test_invalid_or_padded_question_is_rejected(self):
        head = QuestionConditionedSpan(SimpleBase(), 4)
        x = torch.zeros(1, 2, 4)
        mask = torch.tensor([[True, False]])
        for q in (torch.tensor([[False, False]]), torch.tensor([[False, True]])):
            with self.assertRaises(ValueError):
                head(x, mask, q)
