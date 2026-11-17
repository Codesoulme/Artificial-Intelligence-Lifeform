import unittest
import torch
from scripts.deepseek_multitoken_reader_update_audit_20261117 import compare_tensors


class ReaderUpdateAuditTests(unittest.TestCase):
    def test_difference_counts_and_norm(self):
        result = compare_tensors(torch.tensor([1., 2., 3.]), torch.tensor([1., 5., 7.]))
        self.assertEqual((result['changed'], result['total']), (2, 3))
        self.assertAlmostEqual(result['difference_norm'], 5.)
        self.assertAlmostEqual(result['difference_max_abs'], 4.)

    def test_rejects_bad_shapes_and_nonfinite(self):
        with self.assertRaises(ValueError):
            compare_tensors(torch.ones(2), torch.ones(3))
        with self.assertRaises(ValueError):
            compare_tensors(torch.ones(2), torch.tensor([1., float('nan')]))
