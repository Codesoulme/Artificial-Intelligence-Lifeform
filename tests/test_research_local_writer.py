import unittest

import torch
from torch.nn import functional as F
from spora.research_local_writer import FactorizedLocalWriterBank, LocalWriter, LocalWriterBank, calibrate_threshold


class LocalWriterTests(unittest.TestCase):
    def test_calibration_and_selective_correction(self):
        anchor = torch.tensor([1., 0.])
        negative = torch.tensor([[0., 1.], [-1., 0.]])
        threshold = calibrate_threshold(anchor, negative)
        original = torch.eye(2)
        corrected = original * 2
        writer = LocalWriter(original, corrected, anchor, threshold)
        x = torch.stack((anchor, negative[0]))
        self.assertEqual(writer.active(x).tolist(), [True, False])
        torch.testing.assert_close(writer(x)[0], F.linear(x, corrected)[0])
        torch.testing.assert_close(writer(x)[1], F.linear(x, original)[1])

    def test_rejects_indistinguishable_negative(self):
        with self.assertRaisesRegex(ValueError, 'indistinguishable'):
            calibrate_threshold(torch.tensor([1., 0.]), torch.tensor([[2., 0.]]))

    def test_multiple_corrections_and_ambiguous_fallback(self):
        original = torch.eye(2)
        bank = LocalWriterBank(original, [original * 2, original * 3],
                               [torch.tensor([1., 0.]), torch.tensor([0., 1.])], [0.6, 0.6])
        x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., -1.]])
        self.assertEqual(bank.matches(x).tolist(), [[True, False], [False, True], [True, True], [False, False]])
        expected = torch.tensor([[2., 0.], [0., 3.], [1., 1.], [-1., -1.]])
        torch.testing.assert_close(bank(x), expected)

    def test_bank_rejects_mismatched_or_empty_configuration(self):
        with self.assertRaises(ValueError):
            LocalWriterBank(torch.eye(2), [], [], [])
        with self.assertRaises(ValueError):
            LocalWriterBank(torch.eye(2), [torch.eye(2)], [], [0.5])

    def test_factorized_bank_matches_dense_including_collisions(self):
        original = torch.eye(2)
        down = torch.tensor([[[1., 0.]], [[0., 1.]]])
        up = torch.tensor([[[1.], [0.]], [[0.], [2.]]])
        anchors = torch.eye(2)
        thresholds = torch.tensor([0.6, 0.6])
        corrected = [original + up[i] @ down[i] for i in range(2)]
        dense = LocalWriterBank(original, corrected, list(anchors), thresholds.tolist())
        factored = FactorizedLocalWriterBank(original, down, up, anchors, thresholds)
        x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., -1.]])
        self.assertTrue(torch.equal(dense.matches(x), factored.matches(x)))
        torch.testing.assert_close(dense(x), factored(x))


if __name__ == '__main__':
    unittest.main()
