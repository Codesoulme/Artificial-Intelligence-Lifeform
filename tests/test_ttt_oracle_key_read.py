import unittest

import torch

from scripts.deepseek_spora_ttt_oracle_key_read import FixedWriteKey


class OracleKeyReadTests(unittest.TestCase):
    def test_fixed_key_broadcasts_and_rejects_wrong_width(self):
        bridge = FixedWriteKey(torch.tensor([1., 2., 3.]))
        result = bridge(torch.zeros(2, 4, 3))
        torch.testing.assert_close(result, torch.tensor([1., 2., 3.]).expand(2, 4, 3))
        self.assertEqual(bridge.calls, 1)
        with self.assertRaises(ValueError):
            bridge(torch.zeros(1, 2))

    def test_rejects_nonfinite_key(self):
        with self.assertRaises(ValueError):
            FixedWriteKey(torch.tensor([float('nan')]))


if __name__ == '__main__':
    unittest.main()
