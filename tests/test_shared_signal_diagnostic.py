import unittest
import torch
from scripts.deepseek_spora_shared_signal_diagnostic import compare


class SharedSignalDiagnosticTests(unittest.TestCase):
    def test_compare_reports_scale_and_direction(self):
        result = compare(torch.tensor([1., 0.]), torch.tensor([0., 2.]))
        self.assertAlmostEqual(result['first_norm'], 1.)
        self.assertAlmostEqual(result['second_norm'], 2.)
        self.assertAlmostEqual(result['difference_norm'], 5 ** .5, places=6)
        self.assertAlmostEqual(result['cosine'], 0.)


if __name__ == '__main__':
    unittest.main()
