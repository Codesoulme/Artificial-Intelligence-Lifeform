import copy
import tempfile
import unittest
from pathlib import Path

import torch

from spora.durable import TensorStore
from spora.growing import load, new_state, save
from spora.named_address import NamedAddressBinding, address_basis_sha256, feature_files_sha256, structural_feature_sha256
from spora.state_codec import flatten, restore
from spora.temporal import TemporalConfig, TemporalSPORA


class NamedAddressTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6301)
        self.model = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=64,
                                    expert_rank=4, memory_layers=(0, 1),
                                    history_writer='rls', current_address='orthogonal'))
        self.state = new_state(self.model)
        self.digest = 'a' * 64
        self.feature_digest = 'c' * 64
        self.model.named_address_binding = NamedAddressBinding(64, 64, self.digest, 0.9,
            basis_sha256=address_basis_sha256(self.model), feature_config_sha256=self.feature_digest)
        with torch.no_grad():
            self.model.named_address_binding.weights.copy_(torch.eye(64).repeat(2, 1, 1))
        self.features = torch.stack([site.current_basis[7] for site in self.model.memories]).unsqueeze(0)

    def test_codec_portable_and_transaction_roundtrip(self):
        expected = self.model.named_address_binding(self.features, self.model, self.digest, self.feature_digest)
        self.assertEqual(expected['indices'].tolist(), [[7, 7]])
        self.assertEqual(expected['accepted'].tolist(), [True])
        schema, tensors = flatten(self.model, self.state)
        model, state = restore(schema, tensors)
        self.assertEqual(model.named_address_binding(self.features, model, self.digest, self.feature_digest)['indices'].tolist(), [[7, 7]])
        self.assertEqual(flatten(model, state)[0], schema)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'individual.pt'
            save(path, self.model, self.state)
            data = torch.load(path, weights_only=True)
            self.assertEqual(data['format'], 'spora-growing-v2')
            model, state, _ = load(path)
            result = model.named_address_binding(self.features, model, self.digest, self.feature_digest)
            self.assertEqual(result['accepted'].tolist(), [True])
            with TensorStore.create(Path(temp) / 'individual.sqlite', self.model, self.state) as store:
                model, state, _, _ = store.read()
                self.assertEqual(model.named_address_binding(self.features, model, self.digest, self.feature_digest)['indices'].tolist(), [[7, 7]])

    def test_old_model_without_binding_still_uses_legacy_format(self):
        del self.model.named_address_binding
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'legacy.pt'
            save(path, self.model, self.state)
            self.assertEqual(torch.load(path, weights_only=True)['format'], 'spora-growing-v1')
            model, _, _ = load(path)
            self.assertFalse(hasattr(model, 'named_address_binding'))

    def test_mismatch_nonfinite_and_empty_signal_are_rejected(self):
        binding = self.model.named_address_binding
        with self.assertRaises(ValueError):
            binding(self.features, self.model, 'b' * 64, self.feature_digest)
        with self.assertRaises(ValueError):
            binding(self.features, self.model, self.digest, 'd' * 64)
        other = copy.deepcopy(self.model)
        with self.assertRaises(ValueError):
            binding(self.features, other, self.digest, self.feature_digest)
        with self.assertRaises(ValueError):
            binding(torch.full_like(self.features, float('nan')), self.model, self.digest, self.feature_digest)
        with torch.no_grad():
            binding.weights.zero_()
        result = binding(self.features, self.model, self.digest, self.feature_digest)
        self.assertFalse(result['accepted'].item())
        with torch.no_grad():
            binding.weights[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            flatten(self.model, self.state)

    def test_changed_basis_is_rejected_even_on_owning_model(self):
        binding = self.model.named_address_binding
        with torch.no_grad():
            self.model.memories[0].current_basis[0, 0] += 0.1
        with self.assertRaisesRegex(ValueError, 'basis fingerprint'):
            binding(self.features, self.model, self.digest, self.feature_digest)
        with self.assertRaisesRegex(ValueError, 'basis fingerprint'):
            flatten(self.model, self.state)

    def test_span_pooling_contract_survives_reload_and_rejects_mismatch(self):
        binding = self.model.named_address_binding
        binding.feature_contract = 'name-span-mean-v1'
        with self.assertRaisesRegex(ValueError, 'pooling contract'):
            binding(self.features, self.model, self.digest, self.feature_digest)
        schema, tensors = flatten(self.model, self.state)
        model, _ = restore(schema, tensors)
        self.assertEqual(model.named_address_binding.feature_contract, 'name-span-mean-v1')
        with self.assertRaisesRegex(ValueError, 'pooling contract'):
            model.named_address_binding(self.features, model, self.digest, self.feature_digest)
        result = model.named_address_binding(self.features, model, self.digest, self.feature_digest,
                                             feature_contract='name-span-mean-v1')
        self.assertEqual(result['indices'].tolist(), [[7, 7]])
        self.assertEqual(result['accepted'].tolist(), [True])

    def test_legacy_binding_requires_explicit_opt_in(self):
        legacy = NamedAddressBinding(64, 64, self.digest, 0.9)
        with torch.no_grad():
            legacy.weights.copy_(torch.eye(64).repeat(2, 1, 1))
        self.model.named_address_binding = legacy
        schema, tensors = flatten(self.model, self.state)
        model, _ = restore(schema, tensors)
        with self.assertRaisesRegex(ValueError, 'Legacy binding'):
            model.named_address_binding(self.features, model, self.digest)
        result = model.named_address_binding(self.features, model, self.digest, allow_legacy_unbound=True)
        self.assertEqual(result['indices'].tolist(), [[7, 7]])

    def test_feature_files_fingerprint_detects_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
                (root / name).write_text('{}', encoding='utf-8')
            before = feature_files_sha256(root)
            (root / 'tokenizer.json').write_text('{"changed":true}', encoding='utf-8')
            self.assertNotEqual(before, feature_files_sha256(root))

    def test_structural_fingerprint_tracks_layer_weights_and_position_lengths(self):
        layers = {6: '1'*64, 12: '2'*64}
        lengths = {6: 128, 12: 128}
        digest = structural_feature_sha256(self.digest, self.feature_digest, layers, lengths)
        self.assertEqual(digest, structural_feature_sha256(self.digest, self.feature_digest,
                                                           {12: '2'*64, 6: '1'*64}, {12: 128, 6: 128}))
        self.assertNotEqual(digest, structural_feature_sha256(self.digest, self.feature_digest, layers, {6: 768, 12: 128}))
        self.assertNotEqual(digest, structural_feature_sha256(self.digest, self.feature_digest,
                                                              {6: '3'*64, 12: '2'*64}, lengths))
        self.assertNotEqual(digest, structural_feature_sha256(self.digest, self.feature_digest, layers, lengths,
                                                              position_tails={6: '4'*64, 12: None}))
        with self.assertRaises(ValueError):
            structural_feature_sha256(self.digest, self.feature_digest, layers, {6: 128})
        with self.assertRaises(ValueError):
            structural_feature_sha256(self.digest, self.feature_digest, layers, lengths, position_tails={6: '4'*64})

    def test_variant_binding_requires_exact_declared_feature_variant_after_reload(self):
        variant = structural_feature_sha256(self.digest, self.feature_digest, {6: '1'*64}, {6: 128})
        old = self.model.named_address_binding
        with self.assertRaisesRegex(ValueError, 'variant'):
            old(self.features, self.model, self.digest, self.feature_digest, feature_variant_sha256=variant)
        bound = NamedAddressBinding(64, 64, self.digest, 0.9, basis_sha256=address_basis_sha256(self.model),
                                    feature_config_sha256=self.feature_digest, feature_variant_sha256=variant)
        with torch.no_grad():
            bound.weights.copy_(old.weights)
        self.model.named_address_binding = bound
        schema, tensors = flatten(self.model, self.state)
        restored, _ = restore(schema, tensors)
        rebound = restored.named_address_binding
        with self.assertRaisesRegex(ValueError, 'variant'):
            rebound(self.features, restored, self.digest, self.feature_digest)
        with self.assertRaisesRegex(ValueError, 'variant'):
            rebound(self.features, restored, self.digest, self.feature_digest, feature_variant_sha256='f'*64)
        result = rebound(self.features, restored, self.digest, self.feature_digest, feature_variant_sha256=variant)
        self.assertEqual(result['indices'].tolist(), [[7, 7]])

    def test_schema_and_tensor_corruption_are_rejected(self):
        schema, tensors = flatten(self.model, self.state)
        bad_schema = copy.deepcopy(schema)
        bad_schema['named_address_binding']['threshold'] = float('nan')
        with self.assertRaises(ValueError):
            restore(bad_schema, tensors)
        bad_tensors = dict(tensors)
        bad_tensors.pop('model/named_address_binding.weights')
        with self.assertRaises(RuntimeError):
            restore(schema, bad_tensors)


if __name__ == '__main__':
    unittest.main()
