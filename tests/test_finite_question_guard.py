import unittest

from spora.finite_question_guard import valid_finite_question_form


class FiniteQuestionGuardTests(unittest.TestCase):
    def test_canonical_entity_identifiers(self):
        for question in ('对象0的值是多少？', '请告诉我编号 63 目前是多少？', '对象26昨天的值是多少？'):
            self.assertTrue(valid_finite_question_form(question), question)

    def test_unsupported_or_ambiguous_identifiers(self):
        questions = ('查询当前数值。', '对象二十六的值是多少？', '对象-1的值是多少？',
                     '对象8.5的值是多少？', '对象007的值是多少？', '编号100的值是什么？',
                     '对象64的值是多少？', '对象8和对象9的值之和是多少？',
                     '请把对象8的值改成3。', '对象' + '9' * 10000 + '的值是多少？')
        for question in questions:
            self.assertFalse(valid_finite_question_form(question), question[:50])

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            valid_finite_question_form('对象1', True)
        with self.assertRaises(ValueError):
            valid_finite_question_form(None)
