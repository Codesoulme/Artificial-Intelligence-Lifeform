import unittest
from scripts.prepare_squadv2_train_pilot_20261117 import select_train


def row(id_, title, context_en, context, question, impossible, answer=None):
    return {'id': id_, 'title_en': title, 'context_en': context_en, 'context': context,
            'question': question, 'is_impossible': impossible,
            'answers': {'text': [] if answer is None else [answer],
                        'answer_start': [] if answer is None else [context.index(answer)]}}


class SquadTrainSelectionTests(unittest.TestCase):
    def test_excludes_development_title_and_selects_two_kinds(self):
        dev = [row('d', 'held', 'dev text', '小林喝茶。', '谁喝茶？', False, '小林')]
        rows = [row('a', 'held', 'other', '小王喝茶。', '谁喝茶？', False, '小王'),
                row('b', 'train-a', 'a', '小王喝茶。', '谁喝茶？', False, '小王'),
                row('c', 'train-b', 'b', '小赵喝咖啡。', '谁喝茶？', True)]
        chosen, eligible = select_train(rows, dev, per_group=1)
        self.assertEqual(eligible, {'known': 1, 'unknown': 1})
        self.assertEqual([r['kind'] for r in chosen], ['known', 'unknown'])
