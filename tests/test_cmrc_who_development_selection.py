import unittest
from scripts.cmrc_who_development_selection_20261117 import select


class CmrcSelectionTests(unittest.TestCase):
    def test_selection_keeps_only_unique_offset_and_one_context(self):
        paragraph = {'context': '小林喜欢茶，小王喜欢咖啡。', 'qas': [
            {'id': 'a', 'question': '谁喜欢茶？', 'answers': [{'text': '小林', 'answer_start': 0}]},
            {'id': 'b', 'question': '谁喜欢咖啡？', 'answers': [{'text': '小王', 'answer_start': 6}]},
            {'id': 'c', 'question': '谁喜欢茶？', 'answers': [{'text': '小林', 'answer_start': 1}]}]}
        selected, eligible = select({'data': [{'paragraphs': [paragraph]}]})
        self.assertEqual(eligible, 2)
        self.assertEqual(len(selected), 1)
        row = selected[0]
        self.assertIn(row['text'][row['left']:row['right']], ('小林', '小王'))
