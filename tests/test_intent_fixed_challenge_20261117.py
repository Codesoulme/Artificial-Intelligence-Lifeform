import json
import unittest

from scripts.deepseek_intent_fixed_challenge_20261117 import CASES, summarize


class IntentFixedChallengeTests(unittest.TestCase):
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
