import unittest
from scripts.audit_cmrc2018_dev_20261117 import audit


class CmrcAuditTests(unittest.TestCase):
    def test_counts_valid_invalid_absent_and_who(self):
        data = {'data': [{'paragraphs': [{'context': '小林喜欢茶。', 'qas': [
            {'id': 'a', 'question': '谁喜欢茶？', 'answers': [{'answer_start': 0, 'text': '小林'}]},
            {'id': 'b', 'question': '什么饮品？', 'answers': [{'answer_start': 2, 'text': '茶'}]},
            {'id': 'c', 'question': '谁？', 'answers': []}]}]}]}
        result = audit(data)
        self.assertEqual((result['questions'], result['who_question'], result['invalid_offsets']), (3, 2, 1))
        self.assertEqual((result['no_answers'], result['who_all_answers_valid']), (1, 1))

    def test_duplicate_ids_fail_closed(self):
        qa = {'id': 'a', 'question': '谁？', 'answers': []}
        with self.assertRaises(ValueError):
            audit({'data': [{'paragraphs': [{'context': '甲', 'qas': [qa, qa]}]}]})
