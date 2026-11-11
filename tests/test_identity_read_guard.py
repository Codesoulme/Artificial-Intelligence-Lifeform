import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from scripts import deepseek_spora_runtime_chat as runtime
from spora.identity_fingerprints import IdentityFingerprints
from spora.identity_read_guard import IdentityReadGuard


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return '用户:' + messages[0]['content'] + '\n助手:'

    def __call__(self, texts, **kwargs):
        text = texts[0]
        return {'offset_mapping': torch.tensor([[(i, i + 1) for i in range(len(text))]])}


class IdentityReadGuardTests(unittest.TestCase):
    def setUp(self):
        self.fingerprints = IdentityFingerprints(4)
        self.fingerprints.bind(1, '小林')
        self.guard = IdentityReadGuard(self.fingerprints, CharacterTokenizer())

    def mask(self, question, start, end):
        text = self.guard.tokenizer.apply_chat_template([{'content': question}])
        mask = torch.zeros(len(text), dtype=torch.bool)
        mask[3 + start:3 + end] = True
        return mask

    def test_known_name_and_wrong_key(self):
        question = '小林喜欢喝什么？'
        selected = self.mask(question, 0, 2)
        self.assertTrue(self.guard.authorize(question, selected, 1))
        self.assertFalse(self.guard.authorize(question, selected, 2))

    def test_unknown_full_name_rejected_but_truncation_remains_unsafe(self):
        question = '小林华喜欢喝什么？'
        self.assertFalse(self.guard.authorize(question, self.mask(question, 0, 3), 1))
        # This intentional counterexample prevents claiming exact fingerprints
        # can detect a locator that drops part of an unknown person's name.
        self.assertTrue(self.guard.authorize(question, self.mask(question, 0, 2), 1))

    def test_outside_and_discontinuous_selections_rejected(self):
        question = '小林喜欢喝什么？'
        mask = self.mask(question, 0, 2)
        mask[0] = True
        self.assertFalse(self.guard.authorize(question, mask, 1))
        mask = self.mask(question, 0, 1)
        mask[5] = True
        self.assertFalse(self.guard.authorize(question, mask, 1))

    def test_runtime_rejection_precedes_memory_access(self):
        chat = object.__new__(runtime.SporaChat)
        chat.base, chat.tokenizer = None, None
        locator, entity, intent = nn.Linear(4, 1), nn.Linear(4, 1), nn.Linear(4, 1)
        for head in (locator, entity, intent):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, 10)
        temporal = nn.Linear(4, 4)
        nn.init.zeros_(temporal.weight)
        with torch.no_grad():
            temporal.bias.copy_(torch.tensor([10., 0., 0., 0.]))
        chat.router = {'locator_state': locator.state_dict(), 'locator_threshold': 0.5, 'projection': None}
        chat.entity_data = {'gate_state': entity.state_dict(), 'gate_threshold': 0.5, 'address_threshold': 0.5}
        chat.intent_data = {'state': intent.state_dict(), 'threshold': 0.5}
        chat.temporal, chat.temporal_data, chat.bases = temporal, {'threshold': 0.5}, None
        guard = Mock()
        guard.authorize.return_value = False
        features = torch.zeros(1, 2, 3, 2)
        masks = torch.ones(1, 3, dtype=torch.bool)
        with patch.object(runtime, 'extract', return_value=(features, masks)), \
             patch.object(runtime, 'pool', return_value=torch.zeros(1, 4)), \
             patch.object(runtime, 'pooled', return_value=torch.zeros(1, 4)), \
             patch.object(runtime, 'predict', return_value=(torch.tensor([[1, 1]]), torch.ones(1, 2))), \
             patch.object(runtime, 'current_memory_status') as status:
            result = chat.answer('小林喜欢喝什么饮料？', identity_guard=guard)
        self.assertEqual(result['reason'], 'identity_guard_rejected')
        self.assertFalse(result['accepted'])
        guard.authorize.assert_called_once()
        status.assert_not_called()


if __name__ == '__main__':
    unittest.main()
