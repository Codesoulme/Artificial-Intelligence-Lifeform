import unittest
import torch
from spora.research_conditional_null_residual import ConditionalNullResidual


class ConditionalNullTests(unittest.TestCase):
    def test_initial_identity_and_gradient(self):
        torch.manual_seed(42)
        model = ConditionalNullResidual(4, 2)
        x = torch.randn(1, 2, 6)
        self.assertTrue(torch.equal(model(x), x))
        model(x).sum().backward()
        self.assertGreater(float(model.up.weight.grad.norm()), 0.)

    def test_changes_with_input_after_update(self):
        torch.manual_seed(42)
        model = ConditionalNullResidual(4, 2)
        with torch.no_grad():
            model.up.weight.fill_(0.2)
        x = torch.tensor([[[1., 2., 3., 4.], [4., 3., 2., 1.]]])
        delta = model(x) - x
        self.assertFalse(torch.allclose(delta[:, 0], delta[:, 1]))

    def test_rejects_wrong_width(self):
        with self.assertRaises(ValueError):
            ConditionalNullResidual(5)(torch.zeros(1, 2, 4))
