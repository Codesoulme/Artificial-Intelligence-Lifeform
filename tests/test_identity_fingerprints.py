import tempfile
import unittest
from pathlib import Path

import torch

from spora.durable import TensorStore
from spora.growing import load, new_state, save
from spora.identity_fingerprints import IdentityFingerprints
from spora.state_codec import flatten, restore
from spora.temporal import TemporalConfig, TemporalSPORA


class IdentityFingerprintTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(90123)
        torch.set_num_threads(2)
        self.model = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=64,
                                                  expert_rank=4, memory_layers=(0, 1),
                                                  history_writer='rls', current_address='orthogonal'))
        self.state = new_state(self.model)
        self.model.identity_fingerprints = IdentityFingerprints(self.model.config.key_count)

    def test_complete_names_and_near_names_are_distinct(self):
        binding = self.model.identity_fingerprints
        binding.bind(6, '小林')
        self.assertEqual(binding.lookup('小林'), 6)
        for name in ('小林林', '林小林', '小林子', '小林娜'):
            with self.subTest(name=name):
                self.assertIsNone(binding.lookup(name))
        binding.bind(7, '小林子')
        self.assertEqual(binding.lookup('小林子'), 7)
        self.assertEqual(binding.lookup('小林'), 6)

    def test_conflicts_do_not_modify_existing_binding(self):
        binding = self.model.identity_fingerprints
        binding.bind(6, '小林')
        original = binding.digests.clone()
        for key, name in ((6, '阿宁'), (7, '小林')):
            with self.subTest(key=key, name=name), self.assertRaises(ValueError):
                binding.bind(key, name)
        self.assertTrue(torch.equal(binding.digests, original))
        self.assertEqual(binding.lookup('小林'), 6)

    def test_unicode_and_invalid_names(self):
        binding = self.model.identity_fingerprints
        binding.bind(4, 'é')
        self.assertEqual(binding.lookup('e\u0301'), 4)
        for name in ('', ' 小林', '小林 ', '小\x00林', 'x' * 129):
            with self.subTest(name=name), self.assertRaises(ValueError):
                binding.lookup(name)

    def test_tensor_and_file_roundtrips(self):
        self.model.identity_fingerprints.bind(6, '小林')
        schema, tensors = flatten(self.model, self.state)
        restored, restored_state = restore(schema, tensors)
        self.assertEqual(restored.identity_fingerprints.lookup('小林'), 6)
        self.assertIsNone(restored.identity_fingerprints.lookup('小林林'))
        self.assertEqual(restored_state.cursor, self.state.cursor)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'individual.pt'
            save(path, self.model, self.state)
            loaded, _, _ = load(path)
            self.assertEqual(loaded.identity_fingerprints.lookup('小林'), 6)
            database = Path(directory) / 'individual.sqlite'
            with TensorStore.create(database, self.model, self.state) as store:
                loaded, _, _, revision = store.read()
                self.assertEqual(revision, 0)
                self.assertEqual(loaded.identity_fingerprints.lookup('小林'), 6)
                self.assertIsNone(loaded.identity_fingerprints.lookup('小林子'))

    def test_corrupt_identity_state_cannot_be_saved(self):
        binding = self.model.identity_fingerprints
        binding.bind(6, '小林')
        binding.digests[7].copy_(binding.digests[6])
        binding.occupied[7] = True
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            flatten(self.model, self.state)


if __name__ == '__main__':
    unittest.main()
