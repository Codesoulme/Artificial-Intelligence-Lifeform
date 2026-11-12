import tempfile
import unittest
import torch
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import deepseek_spora_runtime_chat as chat_module
from spora.observation_candidate import ObservationCandidate
from spora.growing import new_state
from spora.language_training import frozen_digest
from spora.temporal import TemporalConfig, TemporalSPORA


class RuntimeChatPersistenceTests(unittest.TestCase):
    def test_temporal_constraint_requires_explicit_history_or_later_evidence(self):
        self.assertEqual(chat_module.constrain_temporal_intent('小梅偏爱的饮料是什么？', 1), 0)
        self.assertEqual(chat_module.constrain_temporal_intent('小梅以前喜欢什么饮料？', 0), 1)
        self.assertEqual(chat_module.constrain_temporal_intent('小梅更新后喜欢什么饮料？', 0), 2)
        self.assertEqual(chat_module.constrain_temporal_intent('小梅以前喜欢，后来改成什么？', 0), 3)

    def make_resolution_chat(self):
        chat = object.__new__(chat_module.SporaChat)
        chat.organism = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=64,
                                                    expert_rank=4, memory_layers=(0, 1),
                                                    history_writer='rls', current_address='orthogonal'))
        chat.state = new_state(chat.organism)
        chat.state.cursor = 3
        chat._working_sha256 = None
        chat.decoder = SimpleNamespace(reader=object(), layer_indices=(0, 1), output_adapter=None)
        chat._replacements = lambda: {}
        return chat

    @staticmethod
    def fake_resolution(module, model, state, key, event, *, value, valid_at, decision):
        module.records[0] = torch.tensor([key, event, valid_at, value, 0, 1])
        module.occupied[0] = True

    def test_resolution_save_failure_keeps_original_private(self):
        with tempfile.TemporaryDirectory() as directory:
            chat = self.make_resolution_chat()
            original_model, original_state = chat.organism, chat.state
            digest = frozen_digest(original_model, original_state)
            with patch.object(chat_module, 'WORKING_CHECKPOINT', Path(directory) / 'next.pt'), \
                 patch('spora.historical_resolutions.HistoricalResolutions.record', autospec=True,
                       side_effect=self.fake_resolution), \
                 patch('spora.integrated_checkpoint.save_integrated', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):
                    chat.resolve_historical(1, 2, value=3, valid_at=1, decision='confirm')
            self.assertIs(chat.organism, original_model)
            self.assertIs(chat.state, original_state)
            self.assertFalse(hasattr(chat.organism, 'historical_resolutions'))
            self.assertEqual(frozen_digest(chat.organism, chat.state), digest)

    def test_resolution_publishes_only_after_save(self):
        with tempfile.TemporaryDirectory() as directory:
            chat = self.make_resolution_chat()
            original_model, original_state = chat.organism, chat.state
            original_digest = frozen_digest(original_model, original_state)
            destination = Path(directory) / 'next.pt'

            def save(path, candidate, candidate_state, *args, **kwargs):
                self.assertIs(chat.organism, original_model)
                self.assertIs(chat.state, original_state)
                self.assertIsNot(candidate, original_model)
                self.assertIsNot(candidate_state, original_state)
                self.assertNotEqual(candidate_state.current[0].data_ptr(), original_state.current[0].data_ptr())
                self.assertEqual(candidate.historical_resolutions.lookup(1, 2)['decision'], 'confirm')
                Path(path).write_bytes(b'committed')

            with patch.object(chat_module, 'WORKING_CHECKPOINT', destination), \
                 patch('spora.historical_resolutions.HistoricalResolutions.record', autospec=True,
                       side_effect=self.fake_resolution), \
                 patch('spora.integrated_checkpoint.save_integrated', side_effect=save):
                result = chat.resolve_historical(1, 2, value=3, valid_at=1, decision='confirm')
            self.assertEqual(result['action'], 'historical_resolution_recorded')
            self.assertEqual(chat.organism.historical_resolutions.lookup(1, 2)['value'], 3)
            self.assertEqual(frozen_digest(original_model, original_state), original_digest)
            self.assertEqual(chat._working_sha256, chat_module.sha(destination))

    def test_resolution_stale_checkpoint_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            chat = self.make_resolution_chat()
            destination = Path(directory) / 'next.pt'
            destination.write_bytes(b'newer')
            with patch.object(chat_module, 'WORKING_CHECKPOINT', destination), \
                 patch('spora.historical_resolutions.HistoricalResolutions.record') as record:
                with self.assertRaises(chat_module.RevisionConflict):
                    chat.resolve_historical(1, 2, value=3, valid_at=1, decision='confirm')
                record.assert_not_called()
            self.assertFalse(hasattr(chat.organism, 'historical_resolutions'))
            self.assertEqual(destination.read_bytes(), b'newer')

    def test_candidate_text_must_match_its_finite_value_before_persisting(self):
        chat = object.__new__(chat_module.SporaChat)
        chat.organism = SimpleNamespace(config=SimpleNamespace(key_count=8, value_count=17))
        chat.state = SimpleNamespace(cursor=4)
        candidates = [
            ObservationCandidate(1, '汽水', 6, 'affirmative', 'direct', 1.0, 4, .99, '小林喜欢汽水'),
            ObservationCandidate(1, None, 6, 'affirmative', 'direct', 1.0, 4, .99, ''),
            ObservationCandidate(1, '水', 16, 'affirmative', 'direct', 1.0, 4, .99, '小林喜欢水'),
        ]
        with patch.object(chat, 'observe_sourced') as observe:
            for candidate in candidates:
                with self.subTest(candidate=candidate):
                    result = chat.observe_candidate(candidate)
                    self.assertEqual(result['action'], 'defer')
                    self.assertIn('finite_value_text_mismatch', result['reasons'])
        observe.assert_not_called()

    def test_unresolved_candidate_defers_without_calling_persistence(self):
        chat = object.__new__(chat_module.SporaChat)
        chat.organism = SimpleNamespace(config=SimpleNamespace(key_count=8, value_count=8))
        chat.state = SimpleNamespace(cursor=4)
        candidate = ObservationCandidate(1, '汽水', None, 'affirmative', 'direct', 1.0, 4, .99, '小林喜欢汽水')
        with patch.object(chat, 'observe_sourced') as observe:
            result = chat.observe_candidate(candidate)
        observe.assert_not_called()
        self.assertEqual(result['action'], 'defer')
        self.assertIn('finite_value_unresolved', result['reasons'])

    def test_resolved_candidate_delegates_to_guarded_persistence(self):
        chat = object.__new__(chat_module.SporaChat)
        chat.organism = SimpleNamespace(config=SimpleNamespace(key_count=8, value_count=8))
        chat.state = SimpleNamespace(cursor=4)
        candidate = ObservationCandidate(1, '水', 6, 'affirmative', 'direct', 1.0, 4, .99, '小林喜欢水')
        with patch.object(chat, 'observe_sourced', return_value={'action': 'new'}) as observe:
            result = chat.observe_candidate(candidate)
        observe.assert_called_once_with(1, 6, 'direct', 1.0, 4)
        self.assertEqual(result, {'action': 'new'})

    def test_resolved_candidate_can_use_explicit_calibrated_gate(self):
        chat = object.__new__(chat_module.SporaChat)
        chat.organism = SimpleNamespace(config=SimpleNamespace(key_count=8, value_count=8))
        chat.state = SimpleNamespace(cursor=4)
        candidate = ObservationCandidate(1, '水', 6, 'affirmative', 'direct', .31, 4, .99,
                                         '小林喜欢水')
        with patch.object(chat, 'observe_sourced', return_value={'action': 'new'}) as observe:
            result = chat.observe_candidate(
                candidate,
                calibrated_gate={'policy_id': 'p1', 'authorized': True, 'confidence': .97},
                policy_id='p1')
        observe.assert_called_once_with(1, 6, 'direct', .31, 4)
        self.assertEqual(result, {'action': 'new'})

    def test_calibrated_gate_arguments_must_be_complete(self):
        chat = object.__new__(chat_module.SporaChat)
        chat.organism = SimpleNamespace(config=SimpleNamespace(key_count=8, value_count=8))
        chat.state = SimpleNamespace(cursor=4)
        candidate = ObservationCandidate(1, '水', 6, 'affirmative', 'direct', 1.0, 4, .99,
                                         '小林喜欢水')
        with self.assertRaises(ValueError):
            chat.observe_candidate(candidate, policy_id='p1')

    def test_text_observation_requires_explicit_natural_write_heads(self):
        chat = object.__new__(chat_module.SporaChat)
        chat.natural_write_heads = None
        with self.assertRaisesRegex(ValueError, 'Natural write heads'):
            chat.observe_text('小林喜欢水。')

    def test_text_observation_delegates_parser_result_to_transaction(self):
        chat = object.__new__(chat_module.SporaChat)
        chat.natural_write_heads = object()
        chat.organism = SimpleNamespace(config=SimpleNamespace(key_count=8, value_count=8))
        chat.state = SimpleNamespace(cursor=4)
        candidate = ObservationCandidate(1, '水', 6, 'affirmative', 'direct', 1.0, 4, .99,
                                         '小林喜欢水')
        parsed = {'text': '小林喜欢水。', 'candidate': candidate,
                  'gate': {'policy_id': 'p1', 'authorized': True, 'confidence': .99},
                  'value': '水'}
        with patch('scripts.deepseek_spora_natural_observation_transaction_pilot.build_components',
                   return_value=object()), \
             patch('scripts.deepseek_spora_natural_observation_transaction_pilot.parse_candidate',
                   return_value=parsed), \
             patch.object(chat, 'observe_candidate', return_value={'action': 'new'}) as observe:
            result = chat.observe_text('小林喜欢水。', policy_id='p1')
        observe.assert_called_once()
        self.assertEqual(result['action'], 'new')

    def test_text_observation_cli_requires_explicit_bundle(self):
        # The CLI gate is intentionally represented by the same API contract:
        # no bundle means no natural-language write path.
        chat = object.__new__(chat_module.SporaChat)
        chat.natural_write_heads = None
        with self.assertRaises(ValueError):
            chat.observe_text('小林喜欢水。')

    def test_observation_preserves_bootstrap_and_publishes_after_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bootstrap = root / 'bootstrap.pt'
            bootstrap.write_bytes(b'original')
            destination = root / 'working' / 'individual.pt'
            chat = object.__new__(chat_module.SporaChat)
            chat.state = object()
            chat._working_sha256 = None
            previous = chat.state
            candidate = object()
            chat.organism = object()
            chat.decoder = SimpleNamespace(reader=object(), layer_indices=(0, 1),
                                           output_adapter=object())
            chat._replacements = lambda: {}

            def save(path, organism, state, *args, **kwargs):
                self.assertIs(chat.state, previous)
                self.assertIs(state, candidate)
                Path(path).write_bytes(b'updated')

            with patch.object(chat_module, 'CHECKPOINT', bootstrap), \
                 patch.object(chat_module, 'WORKING_CHECKPOINT', destination), \
                 patch.object(chat_module, 'observe_sourced', return_value=(candidate, {'action': 'new'})), \
                 patch('spora.integrated_checkpoint.save_integrated', side_effect=save):
                chat.observe_sourced(1, 2)
            self.assertEqual(bootstrap.read_bytes(), b'original')
            self.assertEqual(destination.read_bytes(), b'updated')
            self.assertIs(chat.state, candidate)

    def test_failed_save_does_not_publish_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            chat = object.__new__(chat_module.SporaChat)
            chat.state = object()
            chat._working_sha256 = None
            previous = chat.state
            chat.organism = object()
            chat.decoder = SimpleNamespace(reader=object(), layer_indices=(0, 1),
                                           output_adapter=None)
            chat._replacements = lambda: {}
            with patch.object(chat_module, 'WORKING_CHECKPOINT', Path(directory) / 'next.pt'), \
                 patch.object(chat_module, 'observe_sourced', return_value=(object(), {})), \
                 patch('spora.integrated_checkpoint.save_integrated', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):
                    chat.observe_sourced(1, 2)
            self.assertIs(chat.state, previous)

    def test_stale_working_checkpoint_refuses_write(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'individual.pt'
            destination.write_bytes(b'newer state')
            chat = object.__new__(chat_module.SporaChat)
            chat.state = object()
            previous = chat.state
            chat._working_sha256 = None
            with patch.object(chat_module, 'WORKING_CHECKPOINT', destination), \
                 patch.object(chat_module, 'observe_sourced') as observe:
                with self.assertRaises(chat_module.RevisionConflict):
                    chat.observe_sourced(1, 2)
                observe.assert_not_called()
            self.assertIs(chat.state, previous)
            self.assertEqual(destination.read_bytes(), b'newer state')

    def test_generic_preference_does_not_identify_a_beverage(self):
        for question in ('小林的颜色偏好是什么？', '小林的工作偏好是什么？'):
            self.assertFalse(any(word in question for word in chat_module.MEMORY_DOMAIN_ANCHORS))
        self.assertTrue(any(word in '小林喜欢喝什么？' for word in chat_module.MEMORY_DOMAIN_ANCHORS))

    def test_unsupported_and_unknown_values_do_not_reach_output_head(self):
        for value in range(17):
            evidence = torch.zeros(1, 17)
            evidence[0, value] = 10
            self.assertEqual(chat_module._supported_beverage_evidence(evidence, 17), value < 8)
        with self.assertRaises(ValueError):
            chat_module._supported_beverage_evidence(torch.full((1, 17), float('nan')), 17)
        with self.assertRaises(ValueError):
            chat_module._supported_beverage_evidence(torch.zeros(2, 17), 17)


if __name__ == '__main__':
    unittest.main()
