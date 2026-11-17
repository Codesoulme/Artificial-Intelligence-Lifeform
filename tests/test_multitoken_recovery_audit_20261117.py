import unittest
import torch
from scripts.deepseek_multitoken_recovery_audit_20261117 import compare_tree


class RecoveryAuditTests(unittest.TestCase):
    def test_compares_nested_tensors_and_metadata(self):
        left = {'layers': [{'weight': torch.tensor([1., 2.]), 'id': 5}]}
        right = {'layers': [{'weight': torch.tensor([1., 2.]), 'id': 5}]}
        self.assertEqual(compare_tree(left, right), 1)
        right['layers'][0]['weight'][1] = 3.
        with self.assertRaises(ValueError):
            compare_tree(left, right)
        with self.assertRaises(ValueError):
            compare_tree({'x': torch.ones(1)}, {'x': torch.ones(2)})
