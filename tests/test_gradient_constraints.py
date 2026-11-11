import unittest

import torch

from spora.gradient_constraints import project_against_reference


class GradientConstraintTests(unittest.TestCase):
    def test_conflicting_multi_tensor_gradient_is_projected(self):
        gradients = [torch.tensor([-2., 1.]), torch.tensor([0.5])]
        reference = [torch.tensor([1., 0.]), torch.tensor([1.])]
        original = [t.clone() for t in gradients]
        projected, changed = project_against_reference(gradients, reference)
        self.assertTrue(changed)
        self.assertGreaterEqual(sum(float((a * b).sum()) for a, b in zip(projected, reference)), -1e-6)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(gradients, original)))
        self.assertAlmostEqual(float(projected[0][1]), 1.)

    def test_nonconflicting_and_zero_reference_are_unchanged(self):
        for reference in (torch.tensor([1., 0.]), torch.zeros(2)):
            gradient = torch.tensor([1., -2.])
            projected, changed = project_against_reference([gradient], [reference])
            self.assertFalse(changed)
            self.assertTrue(torch.equal(projected[0], gradient))

    def test_nonfinite_input_is_rejected(self):
        with self.assertRaises(ValueError):
            project_against_reference([torch.tensor([float('nan')])], [torch.ones(1)])


if __name__ == '__main__':
    unittest.main()
