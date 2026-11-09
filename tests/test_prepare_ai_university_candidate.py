import unittest

from scripts.prepare_ai_university_candidate import collect_text


class AIUniversityCandidateTests(unittest.TestCase):
    def test_collects_prose_but_excludes_formula_and_provenance(self):
        payload = {'title': '标题', 'text': '正文', 'latex': 'x^2', 'provenance': {'author': 'AI-generated'}, 'practice': [{'problem': '题目', 'numericAnswer': 4}]}
        self.assertEqual(collect_text(payload), ['标题', '正文', '题目'])


if __name__ == '__main__':
    unittest.main()
