import tempfile
import unittest
from pathlib import Path

import torch

from spora.semantic_slot_state import SemanticSlotState, load, save


class SemanticSlotStateTests(unittest.TestCase):
    def test_put_get_and_atomic_roundtrip(self):
        state = SemanticSlotState(4, 2)
        updated = state.put(1, torch.tensor([1., 2., 3., 4.]))
        self.assertEqual(state.revision, 0)
        self.assertEqual(updated.revision, 1)
        with self.assertRaises(KeyError):
            state.get(1)
        torch.testing.assert_close(updated.get(1), torch.tensor([1., 2., 3., 4.]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'slots.pt'
            digest = save(path, updated, base_sha256='base', bridge_sha256='bridge')
            restored, metadata = load(path, expected_base_sha256='base',
                                      expected_bridge_sha256='bridge')
            self.assertEqual(metadata['format'], 'spora-semantic-slot-state-v1')
            self.assertEqual(restored.revision, 1)
            self.assertEqual(digest, metadata.get('integrity_sha256', digest))
            torch.testing.assert_close(restored.get(1), updated.get(1))

    def test_binding_and_nonfinite_values_are_rejected(self):
        state = SemanticSlotState(3, 1)
        with self.assertRaises(ValueError):
            state.put(0, torch.tensor([float('nan'), 1., 2.]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'slots.pt'
            save(path, state, base_sha256='base', bridge_sha256='bridge')
            with self.assertRaisesRegex(ValueError, 'base binding'):
                load(path, expected_base_sha256='other')


if __name__ == '__main__':
    unittest.main()
