import unittest

import torch
from torch import nn

from spora.temporal_agreement import TemporalAgreement


class Fixed(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer('logits', torch.tensor(logits, dtype=torch.float32))

    def forward(self, x):
        return self.logits.expand(x.shape[0], -1)


class TemporalAgreementTests(unittest.TestCase):
    def test_agreement_preserves_baseline_logits(self):
        left = Fixed([0., 4., 0., 0.])
        gate = TemporalAgreement(left, Fixed([0., 3., 0., 0.]), 0.5, 0.5)
        self.assertTrue(torch.equal(gate(torch.zeros(2, 1)), left(torch.zeros(2, 1))))

    def test_disagreement_or_uncertain_head_abstains(self):
        left = Fixed([0., 4., 0., 0.])
        for right, threshold in ((Fixed([0., 0., 4., 0.]), 0.),
                                 (Fixed([0., 0., 0., 4.]), 0.),
                                 (Fixed([0., 1., 0., 0.]), 0.9)):
            gate = TemporalAgreement(left, right, 0.5, threshold)
            self.assertEqual(gate(torch.zeros(1, 1)).argmax(-1).item(), 3)

    def test_invalid_threshold_or_logits_fail(self):
        with self.assertRaises(ValueError):
            TemporalAgreement(Fixed([1., 0., 0., 0.]), Fixed([1., 0., 0., 0.]), float('nan'), 0.)
        gate = TemporalAgreement(Fixed([float('nan'), 0., 0., 0.]), Fixed([1., 0., 0., 0.]), 0., 0.)
        with self.assertRaises(ValueError):
            gate(torch.zeros(1, 1))


if __name__ == '__main__':
    unittest.main()
