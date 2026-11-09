import unittest

import torch

from scripts.deepseek_named_binding_probe import fit_binding, pooled, predict


class NamedBindingProbeTests(unittest.TestCase):
    def test_binding_recovers_orthogonal_examples(self):
        features = torch.eye(4).unsqueeze(1).repeat(1, 2, 1)
        bases = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
        weights = fit_binding(features, features, ridge=0.01)
        indices, scores = predict(features, weights, bases)
        self.assertEqual(indices.tolist(), [[i, i] for i in range(4)])
        self.assertTrue(bool((scores > 0.99).all()))

    def test_invalid_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_binding(torch.zeros(2, 2), torch.zeros(2, 2, 2))
        with self.assertRaises(ValueError):
            pooled(torch.zeros(2, 2, 3, 4), torch.ones(2, 4))


if __name__ == '__main__':
    unittest.main()
