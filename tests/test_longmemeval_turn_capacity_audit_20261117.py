import unittest

from scripts.longmemeval_turn_capacity_audit_20261117 import summarize_lengths


class TurnCapacityTests(unittest.TestCase):
    def test_packet_rounding_and_capacity(self):
        result = summarize_lengths([1, 4, 5, 256, 257])
        self.assertEqual(result['above_one_packet'], 3)
        self.assertEqual(result['median_packets'], 2)
        self.assertEqual(result['over_address_capacity'], 1)

    def test_reject_invalid_lengths(self):
        for lengths in ([], [0], [True], [-1]):
            with self.assertRaises(ValueError):
                summarize_lengths(lengths)
