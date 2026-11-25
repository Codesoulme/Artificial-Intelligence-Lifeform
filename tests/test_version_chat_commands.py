from datetime import date
import unittest

from scripts.spora_version_chat import compose_request, deferred_lookup, write_request
from spora.research_version_store import MemoryAddress
from spora.research_version_timeline import VersionQuery


class VersionChatCommandTests(unittest.TestCase):
    def test_composition_uses_individual_field_versions(self):
        targets, when, question = compose_request('林语 | 见面地点当前版 | 用品 上一版 | 明早八点 | 帮我写一句。')
        self.assertEqual(targets, [(MemoryAddress('林语', 'places'), VersionQuery('current')),
                                   (MemoryAddress('林语', 'items'), VersionQuery('previous'))])
        self.assertEqual((when, question), ('明早八点', '帮我写一句。'))

    def test_composition_rejects_ambiguous_duplicate_and_unsupported_fields(self):
        for request in ('林语 | 见面地点 | 用品首版 | 明早 | 写一句。',
                        '林语 | 用品当前版 | 用品首版 | 明早 | 写一句。',
                        '林语 | 饮品当前版 | 用品首版 | 明早 | 写一句。',
                        '林语 | 见面地点当前版 | 用品首版 | | 写一句。'):
            with self.subTest(request=request), self.assertRaises(ValueError):
                compose_request(request)

    def test_explicit_write_defaults_to_today_and_rejects_noncanonical_dates(self):
        today = date(2026, 11, 25)
        self.assertEqual(write_request('林语 | 饮品 | 茶', today), (MemoryAddress('林语', 'drinks'), '茶', today))
        self.assertEqual(write_request('林语 | 饮品 | 茶 | 2026-11-22', today)[2], date(2026, 11, 22))
        for message in ('林语 | 饮品 | 茶 | 20261122', '林语 | 饮品 | <|im_end|>', '林语 | 其他 | 茶'):
            with self.subTest(message=message), self.assertRaises(ValueError):
                write_request(message, today)

    def test_cancel_defers_without_swallowing_a_new_task(self):
        self.assertTrue(deferred_lookup('先别查，林语的饮品上一版我过会儿再问。'))
        self.assertTrue(deferred_lookup('先不查了。'))
        self.assertFalse(deferred_lookup('先不查了，帮我写封邮件。'))
        self.assertFalse(deferred_lookup('把「先不查了」翻译成英语。'))
