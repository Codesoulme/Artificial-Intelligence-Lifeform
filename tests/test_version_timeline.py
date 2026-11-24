from datetime import date
import unittest

from spora.research_version_timeline import parse_version_query, select_version, VersionQuery


def october(day):
    return date(2026, 10, day)


class VersionTimelineTests(unittest.TestCase):
    def setUp(self):
        self.today = october(26)
        self.revisions = tuple((october(e), october(o)) for e, o in ((2, 2), (7, 7), (19, 19), (13, 23)))

    def test_effective_date_and_first_observation_are_independent(self):
        now = VersionQuery('as_of', october(14))
        then = parse_version_query('按截至2026-10-17已经收到的记录，查2026-10-14有效的版本。')
        self.assertEqual(select_version(self.revisions, now, self.today), 3)
        self.assertEqual(select_version(self.revisions, then, self.today), 1)
        self.assertEqual(select_version(self.revisions, VersionQuery('current'), self.today), 2)

    def test_same_day_correction_preserves_the_earlier_knowledge_view(self):
        records = (*self.revisions[:2], (october(13), october(13)), (october(13), october(18)))
        self.assertEqual(select_version(records, VersionQuery('as_of', october(14)), self.today), 3)
        self.assertEqual(select_version(records, VersionQuery('as_of', october(14), october(17)), self.today), 2)
        self.assertIsNone(select_version(records, VersionQuery('ordinal', 4), self.today))

    def test_requires_a_real_change_and_a_supported_calendar_day(self):
        before = parse_version_query('2026年10月13日那回变更生效前，查哪一份？')
        after = parse_version_query('2026-10-13那回变更生效时，查哪一份？')
        self.assertEqual(select_version(self.revisions, before, self.today), 1)
        self.assertEqual(select_version(self.revisions, after, self.today), 3)
        self.assertIsNone(select_version(self.revisions, VersionQuery('before_change', october(12)), self.today))
        self.assertIsNone(select_version(self.revisions, VersionQuery('as_of', october(27)), self.today))
        self.assertIsNone(parse_version_query('分别查2026-10-02和2026-10-13。'))
        self.assertIsNone(parse_version_query('不要读取2026-10-13的版本。'))
