import unittest
import copy
import tempfile
from pathlib import Path

import torch

from spora.growing import new_state
from spora.durable import TensorStore
from spora.state_codec import flatten, restore
from spora.structured_text_memory import StructuredTextMemoryCodec, StructuredTextMemoryState
from spora.temporal import TemporalConfig, TemporalSPORA


class StructuredTextMemoryTests(unittest.TestCase):
    def setUp(self):
        self.embedding = torch.randn(32, 24)
        self.codec = StructuredTextMemoryCodec(
            self.embedding, memory_width=24, identity_slots=2, fact_slots=4,
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

    def test_codec_config_rebuild_preserves_seeded_codebooks(self):
        code, metadata = self.codec.encode([2], [7, 8])
        rebuilt = StructuredTextMemoryCodec.from_config(
            self.embedding, self.codec.config_dict())
        rebuilt_code, rebuilt_metadata = rebuilt.encode([2], [7, 8])
        decoded = rebuilt.decode(code, metadata)
        self.assertTrue(torch.equal(rebuilt_code, code))
        self.assertEqual(decoded['identity']['token_ids'], [2])
        self.assertEqual(decoded['fact']['token_ids'], [7, 8])
        self.assertEqual(rebuilt.config_dict()['seed'], 71)

    def test_structured_records_round_trip_through_state_codec(self):
        model = TemporalSPORA(TemporalConfig(width=16, layers=2, heads=2,
                                             memory_width=16, expert_rank=4,
                                             memory_layers=(0, 1), history_writer='rls'))
        state = new_state(model)
        config = {'memory_width': 16, 'identity_slots': 1, 'fact_slots': 1}
        structured = StructuredTextMemoryState(config)
        packet = torch.arange(16, dtype=torch.float32)
        state.structured_text = structured.put(
            2, packet, {'identity_length': 1, 'fact_length': 1}, 16,
            model.config.key_count)
        schema, tensors = flatten(model, state)
        self.assertIn('structured_text', schema)
        self.assertIn('structured_text/2', tensors)
        restored_model, restored_state = restore(schema, tensors)
        self.assertEqual(restored_state.structured_text.revision, 1)
        self.assertTrue(torch.equal(restored_state.structured_text.records[2].packet, packet))
        self.assertEqual(restored_state.structured_text.records[2].metadata,
                         {'identity_length': 1, 'fact_length': 1})
        self.assertEqual(restored_model.config.memory_width, 16)

    def test_tampered_structured_metadata_is_rejected(self):
        model = TemporalSPORA(TemporalConfig(width=16, layers=2, heads=2,
                                             memory_width=16, expert_rank=4,
                                             memory_layers=(0, 1), history_writer='rls'))
        state = new_state(model)
        state.structured_text = StructuredTextMemoryState(
            {'memory_width': 16, 'identity_slots': 1, 'fact_slots': 1}).put(
                2, torch.zeros(16), {'identity_length': 1, 'fact_length': 1}, 16,
                model.config.key_count)
        schema, tensors = flatten(model, state)
        tampered = copy.deepcopy(schema)
        tampered['structured_text']['records'][0]['metadata']['fact_length'] = 0
        with self.assertRaises(ValueError):
            restore(tampered, tensors)

    def test_structured_records_use_incremental_durable_transactions(self):
        model = TemporalSPORA(TemporalConfig(width=16, layers=2, heads=2,
                                             memory_width=16, expert_rank=4,
                                             memory_layers=(0, 1), history_writer='rls'))
        state = new_state(model)
        config = {'memory_width': 16, 'identity_slots': 1, 'fact_slots': 1}
        state.structured_text = StructuredTextMemoryState(config).put(
            2, torch.ones(16), {'identity_length': 1, 'fact_length': 1}, 16,
            model.config.key_count)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'structured.sqlite'
            with TensorStore.create(path, model, new_state(model)) as store:
                first = store.commit(model, state, 0)
                self.assertGreaterEqual(first['changed_tensors'], 1)
                loaded_model, loaded_state, _, revision = store.read()
                self.assertEqual(revision, 1)
                self.assertTrue(torch.equal(loaded_state.structured_text.records[2].packet,
                                            torch.ones(16)))
                second_state = loaded_state
                second_state.structured_text = second_state.structured_text.put(
                    3, torch.full((16,), 2.), {'identity_length': 1, 'fact_length': 1},
                    16, loaded_model.config.key_count)
                store.commit(loaded_model, second_state, revision)
                _, latest, _, latest_revision = store.read()
                self.assertEqual(latest_revision, 2)
                self.assertIn(2, latest.structured_text.records)
                self.assertIn(3, latest.structured_text.records)
                failed_state = latest
                failed_state.structured_text = failed_state.structured_text.put(
                    4, torch.full((16,), 3.), {'identity_length': 1, 'fact_length': 1},
                    16, loaded_model.config.key_count)

                def interrupt(stage):
                    if stage == 'after_first_tensor':
                        raise RuntimeError('interrupted')

                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    store.commit(loaded_model, failed_state, latest_revision, hook=interrupt)
                _, recovered, _, recovered_revision = store.read()
                self.assertEqual(recovered_revision, latest_revision)
                self.assertNotIn(4, recovered.structured_text.records)


if __name__ == '__main__':
    unittest.main()
