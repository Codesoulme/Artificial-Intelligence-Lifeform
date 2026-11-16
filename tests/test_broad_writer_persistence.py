import tempfile
import unittest
from pathlib import Path

from scripts.deepseek_spora_broad_writer_persistence import binding
from spora.semantic_phrase_slot_runtime import NaturalSemanticPhraseMemory
from spora.semantic_phrase_slot_state import SemanticPhraseSlotState


class BroadWriterPersistenceTests(unittest.TestCase):
    def test_binding_covers_base_writer_and_each_replacement(self):
        expected = binding('base', 'writer', {'6': 'a', '12': 'b'})
        self.assertEqual(expected, binding('base', 'writer', {'12': 'b', '6': 'a'}))
        self.assertNotEqual(expected, binding('other', 'writer', {'6': 'a', '12': 'b'}))
        self.assertNotEqual(expected, binding('base', 'other', {'6': 'a', '12': 'b'}))
        self.assertNotEqual(expected, binding('base', 'writer', {'6': 'a', '12': 'c'}))

    def test_saved_state_rejects_new_writer_binding(self):
        old = binding('base', 'old-writer', {'6': 'a', '12': 'b'})
        new = binding('base', 'new-writer', {'6': 'a', '12': 'b'})
        memory = NaturalSemanticPhraseMemory(SemanticPhraseSlotState(2, 2), base_sha256='base',
                                             bridge_sha256=old)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'
            memory.save(path)
            NaturalSemanticPhraseMemory.load(path, expected_base_sha256='base', expected_bridge_sha256=old)
            with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                NaturalSemanticPhraseMemory.load(path, expected_base_sha256='base', expected_bridge_sha256=new)


if __name__ == '__main__':
    unittest.main()
