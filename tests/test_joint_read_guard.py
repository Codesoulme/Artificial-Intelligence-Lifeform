import unittest
from unittest.mock import patch

import torch
from torch import nn

from scripts.deepseek_spora_joint_guard_audit import JointSpanReadGuard
from scripts import deepseek_spora_runtime_chat as runtime
from spora.identity_fingerprints import IdentityFingerprints
from spora.identity_read_guard import ConsensusIdentityReadGuard
from tests.test_identity_read_guard import CharacterTokenizer


class FixedHead(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.start = nn.Linear(4, 1)
        self.index = index

    def forward(self, vectors, allowed):
        result = torch.full((1 + len(allowed) ** 2,), -1.0)
        result[self.index] = 1.0
        return result


class JointReadGuardTests(unittest.TestCase):
    def test_runtime_dispatches_feature_guard_before_reading(self):
        fingerprints = IdentityFingerprints(4)
        fingerprints.bind(1, '小林')
        guard = JointSpanReadGuard(fingerprints, CharacterTokenizer(), FixedHead(0))
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
        chat.router = {'locator_state': locator.state_dict(), 'locator_threshold': .5, 'projection': None}
        chat.entity_data = {'gate_state': entity.state_dict(), 'gate_threshold': .5, 'address_threshold': .5}
        chat.intent_data = {'state': intent.state_dict(), 'threshold': .5}
        chat.temporal, chat.temporal_data, chat.bases = temporal, {'threshold': .5}, None
        with patch.object(runtime, 'extract', return_value=(torch.zeros(1, 2, 3, 2), torch.ones(1, 3, dtype=torch.bool))), \
             patch.object(runtime, 'pool', return_value=torch.zeros(1, 4)), \
             patch.object(runtime, 'pooled', return_value=torch.zeros(1, 4)), \
             patch.object(runtime, 'predict', return_value=(torch.tensor([[1, 1]]), torch.ones(1, 2))), \
             patch.object(runtime, 'current_memory_status') as status, \
             patch.object(guard, 'authorize_features', return_value=False) as feature_check:
            result = chat.answer('小林喜欢喝什么饮料？', identity_guard=guard)
        self.assertEqual(result['reason'], 'identity_guard_rejected')
        feature_check.assert_called_once()
        status.assert_not_called()

    def test_veto_mismatch_and_abstention_before_identity(self):
        fingerprints = IdentityFingerprints(4)
        fingerprints.bind(1, '小林')
        question = '小林月喜欢喝什么？'
        length = len('用户:' + question + '\n助手:')
        selected = torch.zeros(length, dtype=torch.bool)
        selected[3:5] = True
        features = torch.zeros(1, 2, length, 2)
        masks = torch.ones(1, length, dtype=torch.bool)
        full_span = 1 + 3 * length + 5
        truncated = 1 + 3 * length + 4
        guard = JointSpanReadGuard(fingerprints, CharacterTokenizer(), FixedHead(full_span))
        self.assertIsInstance(guard, ConsensusIdentityReadGuard)
        self.assertFalse(guard.authorize_features(question, selected, 1, features, masks))
        guard.head.index = 0
        self.assertFalse(guard.authorize_features(question, selected, 1, features, masks))
        guard.head.index = truncated
        # Correlated truncation remains unsafe, despite two agreeing heads.
        self.assertTrue(guard.authorize_features(question, selected, 1, features, masks))
        self.assertFalse(guard.authorize_features(question, selected, 2, features, masks))
        features[0, 0, 0, 0] = float('nan')
        self.assertFalse(guard.authorize_features(question, selected, 1, features, masks))


if __name__ == '__main__':
    unittest.main()
