import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from spora.semantic_phrase_slot_runtime import NaturalSemanticPhraseMemory
from spora.semantic_phrase_slot_state import SemanticPhraseSlotState
from spora.semantic_text_writer import TextSemanticWriter


class SemanticPhraseSlotTests(unittest.TestCase):
    def test_sequence_roundtrip_and_replacement(self):
        state = SemanticPhraseSlotState(3, 2, max_tokens=4)
        memory = NaturalSemanticPhraseMemory(state, base_sha256='base')
        tokenizer = SimpleNamespace(eos_token_id=0, encode=lambda text, **kwargs: [1, 2])
        embedding = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        self.assertEqual(memory.write('甲喜欢喝乙。', tokenizer, embedding)['action'], 'new')
        self.assertEqual(memory.query_slot('甲喜欢喝什么饮料？')['value'].shape, (3, 3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'phrase.pt'
            memory.save(path)
            restored = NaturalSemanticPhraseMemory.load(path, expected_base_sha256='base')
            torch.testing.assert_close(restored.query_slot('甲喜欢喝什么饮料？')['value'],
                                       torch.stack([embedding[1], embedding[2], embedding[0]]))
            self.assertIsNone(restored.query_slot('丙喜欢喝什么饮料？')['slot'])

    def test_encoded_sequence_uses_the_same_routing_and_replaces(self):
        state = SemanticPhraseSlotState(3, 1, max_tokens=4)
        memory = NaturalSemanticPhraseMemory(state, base_sha256='base')
        sequence = torch.ones(2, 3)
        result = memory.write_encoded_sequence('甲喜欢喝乙。', sequence, token_count=1)
        self.assertEqual(result['action'], 'new')
        result = memory.write_encoded_sequence('甲喜欢喝丙。', sequence, token_count=1)
        self.assertEqual(result['action'], 'replace')

    def test_text_writer_preserves_sequence_shape(self):
        writer = TextSemanticWriter(3)
        output = writer(torch.zeros(2, 3))
        self.assertEqual(tuple(output.shape), (2, 3))


if __name__ == '__main__':
    unittest.main()
