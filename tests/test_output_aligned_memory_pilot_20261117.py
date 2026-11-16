import unittest

import torch

from scripts.deepseek_output_aligned_memory_pilot_20261117 import (
    choose_scale, encode_output_rows, make_projection)


class OutputAlignedMemoryTests(unittest.TestCase):
    def test_projection_is_reproducible_and_codes_are_normalized(self):
        first = make_projection(12, 5, 23)
        torch.testing.assert_close(first, make_projection(12, 5, 23))
        codes = encode_output_rows(torch.randn((3, 12)), first)
        torch.testing.assert_close(codes.norm(dim=-1), torch.ones(3))
        with self.assertRaises(ValueError):
            encode_output_rows(torch.zeros((1, 12)), first)

    def test_calibration_ties_choose_smaller_scale(self):
        candidates = [{'scale': 10., 'exact': 5}, {'scale': 1., 'exact': 5},
                      {'scale': 0., 'exact': 4}]
        self.assertEqual(choose_scale(candidates), 1.)
        with self.assertRaises(ValueError):
            choose_scale([])
