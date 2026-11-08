import unittest
from spora.qa_cleanup import strip_qa_labels


class QACleanupTests(unittest.TestCase):
    def test_removes_only_leading_labels(self):
        self.assertEqual(strip_qa_labels('请回答以下问题：\n问题：水是什么？', '答案：水是化合物。'),
                         ('水是什么？', '水是化合物。'))
        self.assertEqual(strip_qa_labels('请解释“答案：”的意思。', '这里的“答案：”是标记。'),
                         ('请解释“答案：”的意思。', '这里的“答案：”是标记。'))
    def test_empty_after_cleanup_is_rejected(self):
        with self.assertRaises(ValueError):
            strip_qa_labels('问题：', '答案：可以')
        with self.assertRaises(ValueError):
            strip_qa_labels('问题：你好', '答：')


if __name__ == '__main__':
    unittest.main()
