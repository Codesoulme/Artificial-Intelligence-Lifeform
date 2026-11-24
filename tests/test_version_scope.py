from datetime import date
import unittest

from spora.identity_fingerprints import IdentityFingerprints
from spora.research_version_scope import addressed_subject, choose_scoped_version
from spora.temporal_routing import CURRENT, PAST


class VersionScopeTests(unittest.TestCase):
    def setUp(self):
        self.today = date(2026, 10, 26)
        self.revisions = ((date(2026, 10, 2), date(2026, 10, 2)),
                          (date(2026, 10, 7), date(2026, 10, 7)))
        self.fingerprints = IdentityFingerprints(2)
        self.fingerprints.bind(0, '霍盈')
        self.fingerprints.bind(1, '范远清')
        self.fields = {0: ('饮品',), 1: ('碰面地点',)}

    def test_explicit_date_and_ordinal_never_fall_back_after_a_failed_lookup(self):
        for text in ('霍盈的饮品，2026-10-01有效的是什么？', '霍盈的饮品第三份记录是什么？'):
            with self.subTest(text=text):
                self.assertEqual(choose_scoped_version(text, self.revisions, self.today, CURRENT),
                                 (None, 'no_matching_version'))
        for text in ('霍盈的饮品2026/10/01当前是什么？', '霍盈的饮品当前第十版是什么？'):
            with self.subTest(text=text):
                self.assertEqual(choose_scoped_version(text, self.revisions, self.today, CURRENT),
                                 (None, 'unresolved_time'))

    def test_fallback_only_uses_current_for_an_unresolved_read(self):
        text = '霍盈的饮品还照哪一项？手头文件上是什么？'
        self.assertEqual(choose_scoped_version(text, self.revisions, self.today, CURRENT), (1, 'learned_current'))
        self.assertEqual(choose_scoped_version(text, self.revisions, self.today, PAST), (None, 'unresolved_time'))
        self.assertEqual(choose_scoped_version('请勿读取霍盈的饮品，告诉我怎么画表。', self.revisions,
                                               self.today, CURRENT), (None, 'outside_lookup_scope'))

    def test_requires_full_name_and_field_boundaries(self):
        self.assertEqual(addressed_subject('请查霍盈的饮品记录是什么？', self.fingerprints, self.fields), 0)
        for text in ('周霍盈的饮品是什么？', '霍盈昭的饮品是什么？', '霍盈的饮品包装是什么？',
                     '霍盈的饮品和范远清的碰面地点是什么？'):
            with self.subTest(text=text):
                self.assertIsNone(addressed_subject(text, self.fingerprints, self.fields))
