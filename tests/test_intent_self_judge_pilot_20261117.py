import unittest
from scripts.deepseek_intent_self_judge_pilot_20261117 import parse_decision, summary


class SelfJudgeTests(unittest.TestCase):
    def test_accepts_only_exact_decisions(self):
        self.assertIs(parse_decision(' 是\n'), True)
        self.assertIs(parse_decision('否'), False)
        self.assertIs(parse_decision('</think>\n\n是的。'), True)
        self.assertIs(parse_decision('不是。'), False)
        for value in ('', '是的但是', '否，但是', '是\n否'):
            self.assertIsNone(parse_decision(value))

    def test_invalid_decisions_are_not_counted_as_correct_rejections(self):
        rows = [{'group': 'x', 'read': False, 'decision': None},
                {'group': 'x', 'read': True, 'decision': True}]
        self.assertEqual(summary(rows)['x'], {'total': 2, 'positive': 1, 'accepted': 1,
                                              'refused': 0, 'invalid': 1, 'correct': 1})
