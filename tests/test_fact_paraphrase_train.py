import unittest
from scripts.deepseek_fact_paraphrase_train_20261117 import build_groups, rewrite


class ParaphraseTrainingTests(unittest.TestCase):
    def test_target_position_survives_question_rewrite(self):
        row = {'text': '小林喜欢喝茶，阿松喜欢蓝色。谁喜欢喝茶？',
               'left': 0, 'right': 2, 'label': 1, 'kind': 'known'}
        for form in ('train', 'unseen'):
            changed = rewrite(row, form)
            self.assertEqual(changed['text'][changed['left']:changed['right']], '小林')
            self.assertNotEqual(changed['text'], row['text'])

    def test_train_and_development_texts_do_not_overlap(self):
        groups = build_groups()
        train = {row['text'] for row in groups['train']}
        self.assertEqual(len(train), len(groups['train']))
        self.assertTrue(any('两个人都叫' in text for text in train))
        for key, rows in groups.items():
            if key != 'train':
                self.assertFalse(train & {row['text'] for row in rows})
