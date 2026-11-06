import unittest
from spora.conversation_metrics import repetition_metrics


class RepetitionTests(unittest.TestCase):
    def test_numbering_must_not_hide_identical_content(self):
        text = '\n'.join(f'{i}. 相同的建议内容' for i in range(1, 8))
        result = repetition_metrics(text)
        self.assertFalse(result['direct_loop'])
        self.assertTrue(result['obvious_loop'])
        self.assertEqual(result['max_repeated_line_count'], 7)

    def test_distinct_list_and_short_answer_are_not_repetition(self):
        self.assertFalse(repetition_metrics('1. 苹果\n2. 香蕉\n3. 草莓\n4. 葡萄')['obvious_loop'])
        self.assertEqual(repetition_metrics('好。')['repeated_4gram_fraction'], 0.)


if __name__ == '__main__':
    unittest.main()
