import tempfile
import subprocess
import sys
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

    def test_fresh_process_loads_saved_slots(self):
        memory = NaturalSemanticPhraseMemory(SemanticPhraseSlotState(3, 2, max_tokens=4),
                                             base_sha256='base')
        memory.write_encoded_sequence('甲喜欢喝乙。', torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'phrase.pt'
            memory.save(path)
            code = (
                'import sys, torch; '
                'from spora.semantic_phrase_slot_runtime import NaturalSemanticPhraseMemory; '
                "m=NaturalSemanticPhraseMemory.load(sys.argv[1], expected_base_sha256='base'); "
                "assert m.state.revision == 1; "
                "assert m.query_slot('甲喜欢喝什么饮料？')['value'].tolist() == [[1.,2.,3.],[4.,5.,6.]]; "
                "assert m.query_slot('乙喜欢喝什么饮料？')['slot'] is None"
            )
            result = subprocess.run([sys.executable, '-c', code, str(path)],
                                    capture_output=True, text=True, encoding='utf-8', timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_chat_failed_save_does_not_publish(self):
        from unittest.mock import patch
        from scripts.deepseek_spora_semantic_chat import persist_fact
        memory = NaturalSemanticPhraseMemory(SemanticPhraseSlotState(3, 2, max_tokens=4),
                                             base_sha256='base')
        tokenizer = SimpleNamespace(eos_token_id=0, encode=lambda text, **kwargs: [1])
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(NaturalSemanticPhraseMemory, 'save', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):
                    persist_fact(memory, '甲喜欢喝乙。', tokenizer, torch.ones(2, 3),
                                 Path(directory) / 'state.pt')
        self.assertEqual(memory.state.revision, 0)
        self.assertIsNone(memory.query_slot('甲喜欢喝什么饮料？')['slot'])

    def test_corrupt_or_wrong_base_checkpoint_is_rejected(self):
        memory = NaturalSemanticPhraseMemory(SemanticPhraseSlotState(3, 1, max_tokens=4),
                                             base_sha256='base')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'phrase.pt'
            memory.save(path)
            with self.assertRaisesRegex(ValueError, 'base binding'):
                NaturalSemanticPhraseMemory.load(path, expected_base_sha256='other')
            payload = torch.load(path, map_location='cpu', weights_only=True)
            payload['tensors']['slots'][0, 0, 0] = 1.
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'integrity'):
                NaturalSemanticPhraseMemory.load(path, expected_base_sha256='base')


if __name__ == '__main__':
    unittest.main()
