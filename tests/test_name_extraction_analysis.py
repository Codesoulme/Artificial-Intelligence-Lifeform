import unittest

from scripts.deepseek_spora_name_extraction_analysis import parse_response, score


class NameExtractionAnalysisTests(unittest.TestCase):
    def test_accepts_only_complete_json_array_or_fence(self):
        self.assertEqual(parse_response('```json\n["小林"]\n```'), ['小林'])
        self.assertEqual(parse_response('[]'), [])
        for text in ('说明\n["小林"]', '```json\n["小林"]', '["小林", "小林"]', '{"name":"小林"}'):
            self.assertIsNone(parse_response(text))

    def test_reports_partial_and_ungrounded_answers(self):
        rows = [{'group': 'x', 'question': '小林舟来了', 'expected': ['小林舟'], 'answer': '["小林"]'},
                {'group': 'x', 'question': '今天下雨', 'expected': [], 'answer': '["小林"]'}]
        result = score(rows)
        self.assertEqual(result['summary']['x'], {'cases': 2, 'parseable': 2, 'grounded': 1, 'exact': 0})


if __name__ == '__main__':
    unittest.main()
