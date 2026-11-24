from datetime import date
import unittest

from spora.identity_fingerprints import IdentityFingerprints
from spora.research_version_clarification import Intent, resolve_version_turn, start_version_turn
from spora.research_version_scope import addressed_subject
from spora.temporal_routing import CURRENT


class VersionClarificationTests(unittest.TestCase):
    def setUp(self):
        self.today = date(2026, 10, 26)
        self.revisions = tuple((date(2026, 10, day), date(2026, 10, day)) for day in (2, 7, 14))
        self.question = '不确定是哪一份。霍盈的饮品写的是什么？'

    def start(self, intent, *, question=None, subject=0, revisions=None):
        return start_version_turn(question or self.question, subject,
                                  self.revisions if revisions is None else revisions, self.today, CURRENT, 0, intent)

    def test_ask_waits_for_one_choice(self):
        ask = self.start(Intent.ASK)
        self.assertIsNone(ask.route)
        self.assertEqual(ask.pending.subject, 0)
        for reply, slot in (('就用现在这份。', 2), ('第一版吧。', 0), ('前一版即可。', 1)):
            with self.subTest(reply=reply):
                resolved = resolve_version_turn(ask.pending, reply, {0: self.revisions}, self.today)
                self.assertEqual(resolved.route, (0, slot))
                self.assertIsNone(resolved.pending)
                self.assertIsNone(resolve_version_turn(resolved.pending, reply, {0: self.revisions}, self.today).route)

    def test_defer_and_new_topic_do_not_read(self):
        deferred = self.start(Intent.DEFER)
        self.assertIsNone(deferred.pending)
        pending = self.start(Intent.ASK).pending
        for reply, expected in (('先不查了。', Intent.DEFER), ('首版或上一版？', Intent.OTHER),
                                ('范远清的碰面地点，首版。', Intent.OTHER), ('把「初版」翻译出来。', Intent.OTHER)):
            with self.subTest(reply=reply):
                result = resolve_version_turn(pending, reply, {0: self.revisions}, self.today)
                self.assertEqual(result.action, expected)
                self.assertIsNone(result.route)
                self.assertIsNone(result.pending)

    def test_transitions_respect_schedule_and_lookup_scope(self):
        self.assertIsNone(self.start(Intent.ASK, subject=None).pending)
        self.assertIsNone(self.start(Intent.ASK, revisions=self.revisions[:1]).pending)
        explicit = '霍盈的饮品，2026-10-07有效的是什么？'
        self.assertEqual(self.start(Intent.READ, question=explicit).route, (0, 1))
        self.assertIsNone(self.start(Intent.ASK, question=explicit).pending)
        quoted = '润色这句。霍盈的饮品写的是什么？'
        self.assertIsNone(self.start(Intent.ASK, question=quoted).pending)
        self.assertIsNone(self.start(Intent.READ, question=quoted).route)
        pending = self.start(Intent.ASK).pending
        self.assertIsNone(resolve_version_turn(pending, '初版', {0: self.revisions[1:]}, self.today).route)

    def test_colloquial_field_boundary_is_scoped_to_exact_subject_and_field(self):
        names = IdentityFingerprints(2)
        names.bind(0, '霍盈')
        names.bind(1, '范远清')
        fields = {0: ('饮品',), 1: ('碰面地点',)}
        question = '霍盈的饮品记的是什么？'
        self.assertIsNone(addressed_subject(question, names, fields))
        self.assertEqual(addressed_subject(question, names, fields, colloquial=True), 0)
        for text in ('周霍盈的饮品记的是什么？', '霍盈的饮品包装记的是什么？',
                     '霍盈的饮品和范远清的碰面地点记的是什么？'):
            with self.subTest(text=text):
                self.assertIsNone(addressed_subject(text, names, fields, colloquial=True))
