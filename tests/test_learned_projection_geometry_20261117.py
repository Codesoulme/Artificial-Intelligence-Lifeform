import unittest
import torch

from scripts.audit_learned_projection_geometry_20261117 import evaluate


class ProjectionGeometryTests(unittest.TestCase):
    def test_duplicate_candidates_are_not_strict_top_one(self):
        train = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        held = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        result = evaluate(train, held, torch.eye(2))
        self.assertGreaterEqual(result['tied_target_rows'], 2)
        self.assertLess(result['strict_top1_within_512'], result['top1_within_512'])
        self.assertEqual(result['strict_top1_within_512'], 1)


if __name__ == '__main__':
    unittest.main()
