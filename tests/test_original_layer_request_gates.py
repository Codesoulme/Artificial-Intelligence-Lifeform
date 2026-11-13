import unittest

from scripts.deepseek_spora_original_layer_name_router import HELD_NEGATIVE
from scripts.deepseek_spora_original_layer_request_gates import build_rows


class OriginalLayerRequestGateDataTests(unittest.TestCase):
    def test_previous_multiple_person_request_keeps_read_but_not_single_label(self):
        prior = {'group': 'no_or_multiple', 'question': HELD_NEGATIVE[2],
                 'guarded_accepted': True, 'guarded_correct_key': False}
        rows = build_rows([prior])
        item = next(row for row in rows if row['split'] == 'prior_name_development')
        self.assertEqual((item['group'], item['single'], item['read']), ('multi', 0, 1))

    def test_previous_unknown_name_is_still_a_single_read_request(self):
        prior = {'group': 'unknown', 'question': '你记得小雪喜欢喝哪种饮品吗？',
                 'guarded_accepted': False, 'guarded_correct_key': False}
        rows = build_rows([prior])
        item = next(row for row in rows if row['split'] == 'prior_name_development')
        self.assertEqual((item['single'], item['read']), (1, 1))

    def test_duplicate_examples_across_splits_are_rejected(self):
        prior = {'group': 'known', 'question': '小林现在喜欢喝哪种饮料？',
                 'guarded_accepted': True, 'guarded_correct_key': True}
        with self.assertRaises(ValueError):
            build_rows([prior])


if __name__ == '__main__':
    unittest.main()
