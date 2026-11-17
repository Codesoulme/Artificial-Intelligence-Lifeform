import unittest
from scripts.prepare_cmrc_train_pilot_20261117 import select_train


def article(title, context, answer, qid):
    return {'title': title, 'paragraphs': [{'context': context, 'qas': [{
        'id': qid, 'question': '谁喜欢茶？',
        'answers': [{'text': answer, 'answer_start': context.index(answer)}]}]}]}


class CmrcTrainSelectionTests(unittest.TestCase):
    def test_excludes_dev_title_and_answer_and_deduplicates_train(self):
        dev = {'data': [article('开发', '小林喜欢茶。', '小林', 'd')]}
        train = {'data': [article('开发', '小宋喜欢茶。', '小宋', 'x'),
                          article('训练一', '小林喜欢茶。', '小林', 'y'),
                          article('训练二', '小王喜欢茶。', '小王', 'a'),
                          article('训练三', '小王喜欢茶。', '小王', 'b')]}
        rows, eligible = select_train(train, dev)
        self.assertEqual(eligible, 2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['text'][rows[0]['left']:rows[0]['right']], '小王')
