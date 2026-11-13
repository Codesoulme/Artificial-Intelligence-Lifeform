import unittest

from scripts.deepseek_spora_explicit_fact_oracle_audit import prompts, strict_answer


class ExplicitFactOracleTests(unittest.TestCase):
    def test_counterfactual_keeps_question_and_changes_fact(self):
        q1, first = prompts('小赵', '喜欢的水果', '水果名称', '山竹')
        q2, second = prompts('小赵', '喜欢的水果', '水果名称', '石榴')
        self.assertEqual(q1, q2)
        self.assertNotIn('山竹', q1)
        self.assertIn('山竹', first)
        self.assertIn('石榴', second)

    def test_strict_scoring_does_not_reward_a_list_or_negation(self):
        self.assertTrue(strict_answer('山竹。', '山竹', '小赵', '喜欢的水果'))
        self.assertTrue(strict_answer('小赵喜欢的水果是山竹。', '山竹', '小赵', '喜欢的水果'))
        self.assertFalse(strict_answer('小赵不喜欢的水果是山竹。', '山竹', '小赵', '喜欢的水果'))
        self.assertFalse(strict_answer('山竹或石榴', '山竹', '小赵', '喜欢的水果'))
        self.assertFalse(strict_answer('不是山竹', '山竹', '小赵', '喜欢的水果'))


if __name__ == '__main__':
    unittest.main()
