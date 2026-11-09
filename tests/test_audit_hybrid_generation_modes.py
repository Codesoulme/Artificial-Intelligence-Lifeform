import unittest

import torch

from scripts.audit_hybrid_generation_modes import nucleus_choice


class GenerationModesAuditTests(unittest.TestCase):
    def test_nucleus_respects_mass_and_seed(self):
        logits = torch.tensor([0.0, -10.0, -20.0])
        rng = torch.Generator().manual_seed(42)
        self.assertEqual(nucleus_choice(logits, rng, 0.9), 0)
        logits = torch.tensor([0.0, 0.0, 0.0])
        a = torch.Generator().manual_seed(123)
        b = torch.Generator().manual_seed(123)
        self.assertEqual([nucleus_choice(logits, a) for _ in range(8)],
                         [nucleus_choice(logits, b) for _ in range(8)])
        with self.assertRaises(ValueError):
            nucleus_choice(logits, rng, 0)


if __name__ == '__main__':
    unittest.main()
