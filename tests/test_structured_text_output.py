import tempfile
import unittest
from pathlib import Path

import torch

from spora.structured_text_output import (StructuredFactSuffixDecoder, StructuredMemoryTextDecoder,
                                          contains_fact_tokens)


class StructuredTextOutputTests(unittest.TestCase):
    def test_fact_token_check_rejects_substitution_and_partial_match(self):
        self.assertTrue(contains_fact_tokens([9, 2, 3, 8], [2, 3]))
        self.assertFalse(contains_fact_tokens([9, 2, 4, 8], [2, 3]))
        self.assertFalse(contains_fact_tokens([2], [2, 3]))
        with self.assertRaises(ValueError):
            contains_fact_tokens([1], [])

    def test_fact_suffix_is_independent_of_identity_lane(self):
        torch.manual_seed(8)
        config = {'memory_width': 22, 'identity_slots': 3, 'identity_slot_width': 4,
                  'fact_slots': 5, 'fact_slot_width': 2}
        decoder = StructuredFactSuffixDecoder(torch.randn(32, 12), config, hidden=16)
        memory = torch.randn(1, 22)
        changed = memory.clone()
        changed[:, :12] = torch.randn(1, 12)
        torch.testing.assert_close(decoder.decode_fact(memory), decoder.decode_fact(changed))

    def test_forward_and_generation_shapes(self):
        torch.manual_seed(9)
        embedding = torch.randn(32, 12)
        decoder = StructuredMemoryTextDecoder(embedding, memory_width=10,
                                              slot_count=5, read_slots=4, hidden=16)
        memory = torch.randn(2, 10)
        targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(tuple(decoder(memory, targets).shape), (2, 3, 32))
        self.assertEqual(decoder.generate(memory, eos_id=31, max_new_tokens=4).shape[0], 2)

    def test_forward_supports_unequal_identity_and_fact_lanes(self):
        torch.manual_seed(10)
        embedding = torch.randn(32, 12)
        decoder = StructuredMemoryTextDecoder(
            embedding, memory_width=22, hidden=16,
            lane_specs=((3, 4), (5, 2)), slot_count=8)
        memory = torch.randn(1, 22)
        targets = torch.tensor([[1, 2]])
        self.assertEqual(tuple(decoder(memory, targets).shape), (1, 2, 32))

    def test_checkpoint_loader_validates_and_rebuilds(self):
        torch.manual_seed(11)
        embedding = torch.randn(32, 12)
        decoder = StructuredMemoryTextDecoder(embedding, memory_width=10,
                                              slot_count=5, read_slots=4, hidden=16)
        payload = {'format': 'spora-identity-codec-composition-v1',
                   'bridge': {k: v.detach().clone() for k, v in decoder.bridge.state_dict().items()},
                   'suffix': {k: v.detach().clone() for k, v in decoder.state_dict().items()},
                   'source_sha256': 's', 'base_sha256': 'b'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'decoder.pt'
            torch.save(payload, path)
            loaded = StructuredMemoryTextDecoder.from_checkpoint(
                embedding, path, expected_source_sha256='s', expected_base_sha256='b')
        self.assertEqual(loaded.embedding_width, 12)
        self.assertEqual(loaded.read_slots, 4)


if __name__ == '__main__':
    unittest.main()
