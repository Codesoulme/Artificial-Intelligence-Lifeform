import unittest

from spora.identity_fingerprints import IdentityFingerprints
from spora.revision_subject import revision_subject


class RevisionSubjectTests(unittest.TestCase):
    def setUp(self):
        self.registry = IdentityFingerprints(8)
        self.registry.bind(1, '小林')
        self.registry.bind(2, '阿明')

    def test_name_before_and_after_the_correction_anchor(self):
        for question in (
            '请查小林第2次修订前记录的饮品。',
            '小林在第三次更新以前登记的是什么？',
            '麻烦查小林第２次调整发生前登记的饮品。',
            '第2次修订完成后，小林被记录为什么？',
            '第三次修订之后，替小林记录的饮品是什么？',
        ):
            with self.subTest(question=question):
                self.assertEqual(revision_subject(self.registry, question),
                                 {'name': '小林', 'key': 1, 'reason': 'enrolled'})

    def test_never_truncates_unregistered_long_names_to_an_enrolled_prefix(self):
        for question in (
            '核对小林璇第1次调整发生前登记的饮品。',
            '第2次修改完成后，小林璇登记的饮品是哪款？',
        ):
            with self.subTest(question=question):
                self.assertEqual(revision_subject(self.registry, question),
                                 {'name': '小林璇', 'key': None, 'reason': 'explicit_name_not_enrolled'})
        self.registry.bind(3, '小林璇')
        self.assertEqual(revision_subject(self.registry, '小林璇第1次修订前喝什么？')['key'], 3)

    def test_refuses_conflicting_subjects_and_quoted_references(self):
        for question, reason in (
            ('小林第2次更正后，阿明的饮品是哪样？', 'ambiguous_revision_subject'),
            ('想知道小林和阿明在第2次修改前采用的饮品。', 'ambiguous_revision_subject'),
            ('翻译「第2次修改后，小林采用的是哪款饮品？」', 'no_single_revision_subject'),
            ('现在小林喜欢什么？', 'no_single_revision_subject'),
        ):
            with self.subTest(question=question):
                self.assertEqual(revision_subject(self.registry, question), {'key': None, 'reason': reason})
        self.registry.bind(3, '李和')
        self.assertEqual(revision_subject(self.registry, '李和第2次修改以前登记的是什么？')['key'], 3)
