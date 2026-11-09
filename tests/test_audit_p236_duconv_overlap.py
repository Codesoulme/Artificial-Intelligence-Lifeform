import unittest

from scripts.audit_p236_duconv_overlap import compare


class OverlapTests(unittest.TestCase):
    def test_full_first_and_long_turn_matches(self):
        long = '共同的话题内容' * 4
        history = [{'messages': [{'content': '你好'}, {'content': long}]}]
        candidate = [{'conversation': ['你好', long]}, {'conversation': ['其他', long]},
                     {'conversation': ['你好', '不同']}, {'conversation': ['不一样', '没有交集']}]
        counts = compare(history, candidate)
        self.assertEqual(counts['candidate_exact_conversations'], 1)
        self.assertEqual(counts['candidate_exact_first_turn'], 2)
        self.assertEqual(counts['candidate_with_exact_turn_at_least_20_chars'], 2)

    def test_rejects_invalid_rows(self):
        with self.assertRaises(ValueError):
            compare([{'messages': []}], [])
        with self.assertRaises(ValueError):
            compare([], [{'conversation': [None]}])


if __name__ == '__main__':
    unittest.main()
