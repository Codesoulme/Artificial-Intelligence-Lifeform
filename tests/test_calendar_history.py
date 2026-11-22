import unittest
from datetime import date

from spora.calendar_history import CalendarFocus, ObservationCalendar, resolve_calendar_followup, resolve_calendar_question
from spora.identity_fingerprints import IdentityFingerprints


class CalendarHistoryTests(unittest.TestCase):
    def setUp(self):
        self.fingerprints = IdentityFingerprints(8)
        self.fingerprints.bind(2, '小林')
        self.payload = {'format': 'spora-observation-calendar-v1',
                        'days': ['2026-01-01', '2026-01-03', '2026-01-03', '2026-01-05']}
        self.calendar = ObservationCalendar.from_payload(self.payload, 4)

    def resolve(self, question):
        return resolve_calendar_question(question, self.fingerprints, self.calendar)

    def test_cutoff_is_inclusive_and_resolves_dates_with_no_observation(self):
        self.assertEqual(self.calendar.locate(date(2026, 1, 2)), 0)
        self.assertEqual(self.calendar.locate(date(2026, 1, 3)), 2)
        self.assertIsNone(self.calendar.locate(date(2026, 1, 6)))

    def test_chinese_date_forms_agree_on_effective_and_observed_time(self):
        questions = (
            '只按截至2026-01-03已收到的记录，2026-01-02对小林有效的饮品是什么？只回答饮品名称。',
            '帮我查小林在2026年1月2日当天登记的饮料是哪种，依据截止到2026-01-03已经掌握的资料？',
        )
        for question in questions:
            self.assertEqual(self.resolve(question), {'reason': 'routed', 'key': 2, 'as_of': 0, 'known_by': 2})
        self.assertIsNone(self.resolve('小林在2026-01-02的饮品是什么？')['known_by'])

    def test_invalid_or_ambiguous_queries_do_not_resolve_a_key(self):
        for question, reason in (
            ('小林在2026年2月30日的饮品是什么？', 'invalid_calendar_date'),
            ('小林在2026-01-06的饮品是什么？', 'date_outside_saved_calendar'),
            ('小林遥在2026-01-02的饮品是什么？', 'date_subject_not_enrolled'),
            ('请翻译「小林在2026-01-02的饮品是什么」', 'unsupported_date_question'),
            ('小林在2026-01-02和2026-01-03的饮品分别是什么？', 'unsupported_date_question'),
        ):
            self.assertEqual(self.resolve(question), {'reason': reason}, question)

    def test_calendar_must_match_observation_count_and_order(self):
        with self.assertRaises(ValueError):
            ObservationCalendar.from_payload(self.payload, 3)
        with self.assertRaises(ValueError):
            ObservationCalendar.from_payload(dict(self.payload, days=list(reversed(self.payload['days']))), 4)

    def test_followup_changes_only_the_asked_coordinate(self):
        focus = CalendarFocus(key=2, as_of=0, known_by=0)
        self.assertEqual(resolve_calendar_followup('那2026年1月3日呢？', self.calendar, focus),
                         {'reason': 'routed', 'key': 2, 'as_of': 2, 'known_by': 0})
        self.assertEqual(resolve_calendar_followup('那按截至2026-01-03已收到的记录呢？', self.calendar, focus),
                         {'reason': 'routed', 'key': 2, 'as_of': 0, 'known_by': 2})
        self.assertEqual(resolve_calendar_followup('那按现在掌握的资料呢？', self.calendar, focus),
                         {'reason': 'routed', 'key': 2, 'as_of': 0, 'known_by': None})

    def test_backfill_followup_uses_the_exact_observation_boundary(self):
        route = {'reason': 'routed', 'key': 2, 'as_of': 0, 'known_by': None}
        selected = {'status': 'experimental_candidate', 'event_time': 2, 'effective_time': 0}
        focus = CalendarFocus.from_read(route, selected)
        self.assertEqual(self.calendar.days[1], self.calendar.days[2])
        followup = '那按补录前的资料呢？'
        self.assertEqual(resolve_calendar_followup(followup, self.calendar, focus)['known_by'], 1)
        switched = CalendarFocus.from_read(dict(route, key=3), None, focus)
        self.assertEqual(resolve_calendar_followup(followup, self.calendar, switched),
                         {'reason': 'date_followup_requires_backfill'})
        self.assertEqual(resolve_calendar_followup(followup, self.calendar, None),
                         {'reason': 'date_followup_requires_subject'})
        self.assertIsNone(CalendarFocus.from_read({'reason': 'unsupported_date_question'}, None, focus))
        self.assertIsNone(resolve_calendar_followup('把「那按补录前的资料呢」翻译成英文', self.calendar, focus))


if __name__ == '__main__':
    unittest.main()
