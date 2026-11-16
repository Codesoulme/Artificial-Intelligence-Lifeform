import unittest
from dataclasses import replace

import torch

from spora.research_chunked_text import decode_chunks, encode_chunks
from spora.structured_text_memory import StructuredTextMemoryCodec


class ChunkedTextTests(unittest.TestCase):
    def setUp(self):
        embedding = torch.randn((32, 64), generator=torch.Generator().manual_seed(42))
        self.codec = StructuredTextMemoryCodec(embedding, memory_width=90, identity_slots=3,
                                               fact_slots=5, identity_slot_width=15,
                                               fact_slot_width=9, eos_token_id=0)

    def test_roundtrip_multiple_chunks(self):
        fact = list(range(2, 13))
        chunks = encode_chunks(self.codec, [1], fact, max_chunks=3)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(decode_chunks(self.codec, chunks, max_chunks=3),
                         {'identity_ids': [1], 'fact_ids': fact})

    def test_rejects_truncation_reorder_and_mixed_identity(self):
        chunks = encode_chunks(self.codec, [1], list(range(2, 11)), max_chunks=3)
        with self.assertRaises(ValueError):
            decode_chunks(self.codec, chunks[:-1], max_chunks=3)
        with self.assertRaises(ValueError):
            decode_chunks(self.codec, (chunks[1], chunks[0], chunks[2]), max_chunks=3)
        other = encode_chunks(self.codec, [2], list(range(2, 11)), max_chunks=3)
        with self.assertRaises(ValueError):
            decode_chunks(self.codec, (chunks[0], other[1], chunks[2]), max_chunks=3)
        with self.assertRaises(ValueError):
            decode_chunks(self.codec, (replace(chunks[0], total=5), *chunks[1:]), max_chunks=3)

    def test_capacity_and_invalid_tokens_fail_before_output(self):
        with self.assertRaises(ValueError):
            encode_chunks(self.codec, [1], list(range(2, 11)), max_chunks=2)
        with self.assertRaises(ValueError):
            encode_chunks(self.codec, [1], [2, 0, 3], max_chunks=3)
        with self.assertRaises(ValueError):
            encode_chunks(self.codec, [1], [], max_chunks=3)
