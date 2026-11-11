import tempfile
import unittest
import torch
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import deepseek_spora_runtime_chat as chat_module


class RuntimeChatPersistenceTests(unittest.TestCase):
    def test_observation_preserves_bootstrap_and_publishes_after_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bootstrap = root / 'bootstrap.pt'
            bootstrap.write_bytes(b'original')
            destination = root / 'working' / 'individual.pt'
            chat = object.__new__(chat_module.SporaChat)
            chat.state = object()
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
