import unittest

from spora.chat import bounded_conversation_prompt


class ChatContextTests(unittest.TestCase):
    def test_newest_turns_fit_and_oldest_turns_are_dropped(self):
        history = [('甲', '答甲'), ('乙', '答乙'), ('丙', '答丙')]
        prompt = bounded_conversation_prompt(history, '丁', 30, max_turns=3)
        self.assertIn('用户：丁', prompt)
        self.assertNotIn('用户：甲', prompt)

    def test_empty_history_is_a_single_current_turn(self):
        self.assertEqual(bounded_conversation_prompt([], '你好', 100), '用户：你好\n助手：')


if __name__ == '__main__':
    unittest.main()
