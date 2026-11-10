import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from spora.growing import new_state
from spora.integrated_checkpoint import (_digest, IntegratedSPORARuntime, load_integrated,
                                         load_integrated_runtime, save_integrated)
from spora.integrated_open_base import IntegratedOpenBase, NativeSPORAMemoryReader, ScaledOpenBaseMemory
from spora.internal_memory import WriteControl, attach, read_provenance
from spora.language_training import frozen_digest
from spora.named_address import NamedAddressBinding, address_basis_sha256, feature_files_sha256
from spora.temporal import TemporalConfig, TemporalSPORA
from tests.test_integrated_open_base import Base


class IntegratedCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'model.safetensors').write_bytes(b'base weights')
        (self.root / 'config.json').write_text('{"hidden_size":3,"num_hidden_layers":3}', encoding='utf-8')
        for name in ('tokenizer.json', 'tokenizer_config.json'):
            (self.root / name).write_text('{}', encoding='utf-8')
        self.base_sha = hashlib.sha256(b'base weights').hexdigest()
        self.config_sha = feature_files_sha256(self.root)
        self.organism = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=64,
                                      expert_rank=4, memory_layers=(0, 1),
                                      history_writer='rls', current_address='orthogonal'))
        self.organism.named_address_binding = NamedAddressBinding(
            64, 64, self.base_sha, 0.8, feature_contract='name-span-mean-v1',
            basis_sha256=address_basis_sha256(self.organism), feature_config_sha256=self.config_sha)
        self.state = new_state(self.organism)
        self.state.current = [torch.eye(64).unsqueeze(0) for _ in range(2)]
        self.reader = ScaledOpenBaseMemory(3, 64, 0.5)
        with torch.no_grad():
            for projection in self.reader.out:
                projection.weight[:, :3].copy_(torch.eye(3))
            self.reader.gate.zero_()
        self.path = self.root / 'individual.pt'

    def tearDown(self):
        self.temp.cleanup()

    def base(self):
        model = Base()
        model.config = SimpleNamespace(hidden_size=3)
        return model

    def save(self):
        save_integrated(self.path, self.organism, self.state, self.reader, self.root, (0, 2))

    def test_roundtrip_preserves_model_state_and_outputs(self):
        x = torch.ones(1, 1, 3)
        addresses = [torch.ones(1, 64), torch.ones(1, 64)]
        reference = IntegratedOpenBase(self.base(), self.reader, (0, 2))
        expected = reference(x, memory_state=self.state, memory_addresses=addresses, current_only=True)
        self.save()
        model, organism, state = load_integrated(self.path, self.base(), self.root)
        self.assertEqual(frozen_digest(organism, state), frozen_digest(self.organism, self.state))
        self.assertEqual(organism.named_address_binding.feature_contract, 'name-span-mean-v1')
        self.assertTrue(torch.equal(model(x, memory_state=state, memory_addresses=addresses, current_only=True), expected))

    def test_native_spora_reader_roundtrip_preserves_outputs(self):
        legacy = self.reader
        native = NativeSPORAMemoryReader(3, 64, 0.5)
        native.initialize_from_legacy(legacy)
        self.reader = native
        x = torch.randn(1, 2, 3)
        addresses = [torch.randn(1, 64), torch.randn(1, 64)]
        reference = IntegratedOpenBase(self.base(), native, (0, 2))
        expected = reference(x, memory_state=self.state, memory_addresses=addresses, current_only=True)
        self.save()
        payload = torch.load(self.path, weights_only=True)
        self.assertEqual(payload['metadata']['format'], 'spora-integrated-open-base-v2')
        self.assertEqual(payload['metadata']['reader']['kind'], 'native-spora-delta-site')
        model, organism, state = load_integrated(self.path, self.base(), self.root)
        self.assertIsInstance(model.reader, NativeSPORAMemoryReader)
        self.assertEqual(frozen_digest(organism, state), frozen_digest(self.organism, self.state))
        actual = model(x, memory_state=state, memory_addresses=addresses, current_only=True)
        self.assertTrue(torch.equal(actual, expected))

    def test_v1_checkpoint_still_loads_as_legacy_reader(self):
        self.save()
        payload = torch.load(self.path, weights_only=True)
        payload['metadata']['format'] = 'spora-integrated-open-base-v1'
        payload['metadata']['reader'].pop('kind')
        payload['integrity_sha256'] = _digest(payload['metadata'], payload['tensors'])
        torch.save(payload, self.path)
        model, _, _ = load_integrated(self.path, self.base(), self.root)
        self.assertIs(type(model.reader), ScaledOpenBaseMemory)

    def test_integrated_runtime_writes_corrects_and_recovers_atomically(self):
        controller = WriteControl()
        with torch.no_grad():
            for parameter in controller.parameters():
                parameter.zero_()
            controller.network[-1].bias[0] = 20
        self.state = attach(self.organism, self.state, controller)
        native = NativeSPORAMemoryReader(3, 64, 0.5)
        native.initialize_from_legacy(self.reader)
        save_integrated(self.path, self.organism, self.state, native, self.root, (0, 2))
        runtime = load_integrated_runtime(self.path, self.base(), self.root)
        self.assertIsInstance(runtime, IntegratedSPORARuntime)
        initial_digest = frozen_digest(runtime.organism, runtime.state)
        with self.assertRaises(OSError):
            runtime.observe_sourced(4, 2, 'direct', 1.0, checkpoint_path=self.root / 'missing' / 'state.pt')
        self.assertEqual(frozen_digest(runtime.organism, runtime.state), initial_digest)
        self.assertEqual(runtime.state.cursor, 0)

        first = runtime.observe_sourced(4, 2, 'direct', 1.0)
        self.assertTrue(first['current_write'])
        self.assertEqual(runtime.state.cursor, 1)
        key = torch.tensor([4])
        modes = torch.ones_like(key)
        _, old_bound = runtime.organism.representation(key, modes, torch.tensor([0]))
        native_site = runtime.organism.memories[-1]
        old_address = runtime.organism.memory_address(native_site, old_bound, key, modes)
        old_read_before_correction = native_site.retrieve([runtime.state.pages[0][-1][0]], old_address).clone()
        with torch.no_grad():
            for parameter in runtime.organism.write_control.parameters():
                parameter.zero_()
            runtime.organism.write_control.network[-1].bias[1] = 20
        corrected = runtime.observe_sourced(4, 5, 'direct', 1.0)
        self.assertTrue(corrected['current_write'])
        self.assertEqual(runtime.state.cursor, 2)

        recovered = load_integrated_runtime(self.path, self.base(), self.root)
        self.assertEqual(recovered.state.cursor, 2)
        key = torch.tensor([4])
        modes = torch.ones_like(key)
        for event_index, expected_action in ((0, 'new'), (1, 'correct')):
            self.assertEqual(read_provenance(recovered.organism, recovered.state, 4, event_index)['action'],
                             expected_action)
            event_time = torch.tensor([event_index])
            _, bound = recovered.organism.representation(key, modes, event_time)
            site = recovered.organism.memories[-1]
            address = recovered.organism.memory_address(site, bound, key, modes)
            retrieved = site.retrieve([recovered.state.pages[0][-1][0]], address)
            self.assertTrue(torch.isfinite(retrieved).all())
        self.assertFalse(torch.allclose(retrieved, old_read_before_correction))

    def test_wrong_base_or_configuration_is_rejected(self):
        self.save()
        (self.root / 'model.safetensors').write_bytes(b'different')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            load_integrated(self.path, self.base(), self.root)
        (self.root / 'model.safetensors').write_bytes(b'base weights')
        (self.root / 'tokenizer.json').write_text('{"changed":true}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            load_integrated(self.path, self.base(), self.root)

    def test_corruption_and_invalid_save_preserve_previous_checkpoint(self):
        self.save()
        original = self.path.read_bytes()
        with torch.no_grad():
            self.reader.gate[0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            self.save()
        self.assertEqual(self.path.read_bytes(), original)
        payload = torch.load(self.path, weights_only=True)
        payload['tensors']['reader/gate'][0] += 1
        torch.save(payload, self.path)
        with self.assertRaisesRegex(ValueError, 'integrity'):
            load_integrated(self.path, self.base(), self.root)

    def test_wrong_binding_and_base_width_are_rejected(self):
        self.organism.named_address_binding.feature_model_sha256 = '0' * 64
        with self.assertRaisesRegex(ValueError, 'binding'):
            self.save()
        self.organism.named_address_binding.feature_model_sha256 = self.base_sha
        self.save()
        base = self.base()
        base.config.hidden_size = 4
        with self.assertRaisesRegex(ValueError, 'width'):
            load_integrated(self.path, base, self.root)

    def test_invalid_layer_indices_do_not_overwrite_checkpoint(self):
        self.save()
        previous = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'base configuration'):
            save_integrated(self.path, self.organism, self.state, self.reader, self.root, (0, 8))
        self.assertEqual(self.path.read_bytes(), previous)

    def test_cannot_replace_base_weight_or_configuration(self):
        for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json'):
            path = self.root / name
            previous = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'protected base'):
                save_integrated(path, self.organism, self.state, self.reader, self.root, (0, 2))
            self.assertEqual(path.read_bytes(), previous)


if __name__ == '__main__':
    unittest.main()
