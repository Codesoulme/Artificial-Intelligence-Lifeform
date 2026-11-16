import unittest

import torch
from torch.nn import functional as F
from spora.research_local_writer import LocalWriter, calibrate_threshold


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


if __name__ == '__main__':
    unittest.main()
