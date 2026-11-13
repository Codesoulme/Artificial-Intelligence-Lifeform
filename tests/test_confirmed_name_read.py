import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from scripts import deepseek_spora_runtime_chat as runtime
from spora.confirmed_name_read import ConfirmedNameGuard, answer_with_confirmed_name
from spora.identity_fingerprints import IdentityFingerprints
from tests.test_identity_read_guard import CharacterTokenizer


class ConfirmedNameReadTests(unittest.TestCase):
    def setUp(self):
        self.names = IdentityFingerprints(4)
        self.names.bind(1, '小林')
        self.tokenizer = CharacterTokenizer()

    def selected(self, question, name):
        text = self.tokenizer.apply_chat_template([{'content': question}])
        mask = torch.zeros(len(text), dtype=torch.bool)
        start = 3 + question.index(name)
        mask[start:start + len(name)] = True
        return mask

    def test_no_confirmation_or_unknown_never_calls_chat(self):
        chat = SimpleNamespace(answer=Mock(), tokenizer=self.tokenizer,
                               organism=SimpleNamespace(identity_fingerprints=self.names))
        self.assertEqual(answer_with_confirmed_name(chat, '小林喜欢什么？')['reason'], 'explicit_name_required')
        self.assertEqual(answer_with_confirmed_name(chat, '小林舟喜欢什么？', '小林舟')['reason'],
                         'explicit_name_not_enrolled')
        chat.answer.assert_not_called()

    def test_confirmation_is_bound_to_full_text_question_and_key(self):
        question = '小林舟喜欢喝什么？'
        selected = self.selected(question, '小林')
        guard = ConfirmedNameGuard(self.names, self.tokenizer, question, '小林舟')
        self.assertFalse(guard.authorize(question, selected, 1))
        guard = ConfirmedNameGuard(self.names, self.tokenizer, question, '小林')
        self.assertFalse(guard.authorize('小林舟喜欢什么？', selected, 1))
        self.assertFalse(guard.authorize(question, selected, 2))
        # A mistaken user-supplied short name can still confirm a truncation.
        self.assertTrue(guard.authorize(question, selected, 1))

    def test_runtime_blocks_mismatch_before_memory_status(self):
        chat = object.__new__(runtime.SporaChat)
        chat.base, chat.tokenizer = None, self.tokenizer
        chat.organism = SimpleNamespace(identity_fingerprints=self.names)
        locator, entity, intent = nn.Linear(4, 1), nn.Linear(4, 1), nn.Linear(4, 1)
        for head in (locator, entity, intent):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, 10)
        temporal = nn.Linear(4, 4)
        nn.init.zeros_(temporal.weight)
        with torch.no_grad():
            temporal.bias.copy_(torch.tensor([10., 0., 0., 0.]))
        chat.router = {'locator_state': locator.state_dict(), 'locator_threshold': .5, 'projection': None}
        chat.entity_data = {'gate_state': entity.state_dict(), 'gate_threshold': .5, 'address_threshold': .5}
        chat.intent_data = {'state': intent.state_dict(), 'threshold': .5}
        chat.temporal, chat.temporal_data, chat.bases, chat.state = temporal, {'threshold': .5}, None, None
        question = '小林舟喜欢喝什么饮料？'
        selected = self.selected(question, '小林')
        self.names.bind(2, '小林舟')
        n = len(selected)
        with patch.object(runtime, 'extract', return_value=(torch.zeros(1, 2, n, 2), torch.ones(1, n, dtype=torch.bool))), \
             patch.object(runtime, 'pool', return_value=torch.zeros(1, 4)), \
             patch.object(runtime, 'pooled', return_value=torch.zeros(1, 4)), \
             patch.object(runtime, 'predict', return_value=(torch.tensor([[1, 1]]), torch.ones(1, 2))), \
             patch.object(runtime, 'single_span_mask', return_value=(selected, True)), \
             patch.object(runtime, 'current_memory_status', return_value={'status': 'unknown'}) as status:
            result = answer_with_confirmed_name(chat, question, '小林舟')
            self.assertEqual(result['reason'], 'identity_guard_rejected')
            status.assert_not_called()
            result = answer_with_confirmed_name(chat, question, '小林')
            self.assertEqual(result['reason'], 'unknown_memory')
            status.assert_called_once()


if __name__ == '__main__':
    unittest.main()
