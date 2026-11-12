import unittest

import torch

from spora.structured_text_memory import StructuredTextMemoryCodec


class StructuredTextMemoryTests(unittest.TestCase):
    def setUp(self):
        embedding = torch.randn(32, 24)
        self.codec = StructuredTextMemoryCodec(
            embedding, memory_width=24, identity_slots=2, fact_slots=4,
            seed=71, eos_token_id=31)

    def test_lanes_round_trip_independently(self):
        code, metadata = self.codec.encode([2], [7, 8])
        decoded = self.codec.decode(code, metadata)
        self.assertEqual(decoded['identity']['token_ids'], [2])
        self.assertEqual(decoded['fact']['token_ids'], [7, 8])
        self.assertEqual(decoded['metadata'], {'identity_length': 1, 'fact_length': 2})

    def test_identity_and_fact_boundaries_are_explicit(self):
        code, metadata = self.codec.encode([2], [7, 8])
        changed = dict(metadata, fact_length=1)
        with self.assertRaises(ValueError):
            self.codec.decode(code, changed)

    def test_lane_capacity_is_independent(self):
        with self.assertRaisesRegex(ValueError, 'Identity'):
            self.codec.encode([2, 3, 4], [7])
        with self.assertRaisesRegex(ValueError, 'Fact'):
            self.codec.encode([2], [7, 8, 9, 10])

    def test_metadata_is_required_and_validated(self):
        code, metadata = self.codec.encode([2], [7])
        with self.assertRaises(ValueError):
            self.codec.decode(code, None)
        with self.assertRaises(ValueError):
            self.codec.decode(code, dict(metadata, identity_length=0))

    def test_unequal_lane_widths_round_trip(self):
        codec = StructuredTextMemoryCodec(
            torch.randn(32, 24), memory_width=24, identity_slots=2, fact_slots=2,
            identity_slot_width=8, fact_slot_width=4, seed=73, eos_token_id=31)
        code, metadata = codec.encode([2], [7])
        decoded = codec.decode(code, metadata)
        self.assertEqual(decoded['identity']['token_ids'], [2])
        self.assertEqual(decoded['fact']['token_ids'], [7])


if __name__ == '__main__':
    unittest.main()
