import json
import unittest

from types import SimpleNamespace
import torch
from torch.nn import functional as F
from scripts.deepseek_intent_fixed_challenge_20261117 import CASES, extract_feature, summarize


class IntentFixedChallengeTests(unittest.TestCase):
    def test_feature_selection_uses_requested_layer_and_pool(self):
        early = torch.tensor([[[1., 0.], [1., 2.]]])
        final = torch.tensor([[[0., 3.], [4., 0.]]])
        output = SimpleNamespace(hidden_states=(None, early), last_hidden_state=final)
        torch.testing.assert_close(extract_feature(output, 'early_mean'), F.normalize(torch.tensor([1., 1.]), dim=0))
        torch.testing.assert_close(extract_feature(output, 'final_mean'), F.normalize(torch.tensor([2., 1.5]), dim=0))
        torch.testing.assert_close(extract_feature(output, 'final_last'), torch.tensor([1., 0.]))
        with self.assertRaises(ValueError):
            extract_feature(output, 'unknown')
    def test_group_accounting_separates_false_accepts_and_rejects(self):
        rows = [
            {'group': 'current', 'read': True, 'accepted': True},
            {'group': 'current', 'read': True, 'accepted': False},
            {'group': 'attribute', 'read': False, 'accepted': True},
            {'group': 'attribute', 'read': False, 'accepted': False},
        ]
        scores = summarize(rows)
        self.assertEqual(scores['current'], {'total': 2, 'accepted': 1, 'expected_accept': 2, 'correct': 1})
        self.assertEqual(scores['attribute'], {'total': 2, 'accepted': 1, 'expected_accept': 0, 'correct': 1})

    def test_challenge_has_balanced_groups_and_unique_questions(self):
        rows = json.loads(CASES.read_text(encoding='utf-8'))
        self.assertEqual(len({r['question'] for r in rows}), 36)
        for group in ('current', 'past', 'attribute', 'other_relation', 'non_read', 'ordinary'):
            members = [r for r in rows if r['group'] == group]
            self.assertEqual(len(members), 6)
            self.assertTrue(all(r['read'] == (group in ('current', 'past')) for r in members))
