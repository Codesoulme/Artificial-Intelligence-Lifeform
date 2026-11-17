import unittest
import torch

from scripts.deepseek_reviewed_span_boundary_audit_20261117 import diagnose


class BoundaryAuditTests(unittest.TestCase):
    def test_exact_unaligned_and_width(self):
        offsets = torch.tensor([[0, 0], [10, 11], [11, 12], [12, 13], [0, 0]])
        allowed = torch.tensor([False, True, True, True, False])
        prediction = {'linear': {'selected': False}, 'context': {'selected': False},
                      'consensus': {'selected': False, 'exact': False}}
        row = {'text': '甲乙丙', 'left': 0, 'right': 2, 'digest': 'example'}
        exact = diagnose(row, offsets, allowed, 10, prediction)
        self.assertTrue(exact['usable'] and exact['tokens'] == 2 and exact['category'] == 'both_reject')
        crossing = offsets.clone()
        crossing[2] = torch.tensor([11, 13])
        self.assertFalse(diagnose(row, crossing, allowed, 10, prediction)['exact_boundary'])
        long_offsets = torch.tensor([[i, i + 1] for i in range(13)])
        long_row = {'text': '甲' * 13, 'left': 0, 'right': 13, 'digest': 'long'}
        too_long = diagnose(long_row, long_offsets, torch.ones(13, dtype=torch.bool), 0, prediction)
        self.assertTrue(too_long['exact_boundary'] and not too_long['usable'])
