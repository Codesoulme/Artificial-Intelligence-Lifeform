import unittest
from scripts.deepseek_cmrc_question_first_probe_20261117 import question_first
from scripts.deepseek_cmrc_question_first_training_20261117 import ordered_extract
from unittest.mock import patch


class QuestionFirstTests(unittest.TestCase):
    def test_gold_span_tracks_moved_context(self):
        row = {'text': '小林喜欢茶。\n问题：谁喜欢茶？', 'left': 0, 'right': 2}
        result = question_first(row)
        self.assertTrue(result['text'].startswith('问题：谁喜欢茶？\n资料：'))
        self.assertEqual(result['text'][result['left']:result['right']], '小林')

    def test_rejects_bad_bounds(self):
        with self.assertRaises(ValueError):
            question_first({'text': '甲。\n问题：谁？', 'left': 3, 'right': 4})

    def test_replay_keeps_original_format(self):
        natural = {'text': '小林喜欢茶。\n问题：谁喜欢茶？', 'left': 0, 'right': 2}
        replay = {'text': '小林喜欢茶。谁喜欢茶？', 'left': 0, 'right': 2}
        with patch('scripts.deepseek_cmrc_question_first_training_20261117.ORIGINAL_EXTRACT',
                   side_effect=lambda chat, rows: rows):
            result = ordered_extract(None, [natural, replay])
        self.assertEqual(result[1], replay)
        self.assertTrue(result[0]['text'].startswith('问题：'))
