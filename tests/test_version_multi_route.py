from datetime import date
import unittest

from spora.research_multi_version_route import Lookup, explicit_queries, mentions, resolve
from spora.research_version_store import MemoryAddress, MemoryVersion
from spora.research_version_timeline import VersionQuery


FIELDS = {'drink': '饮品', 'place': '见面地点'}
NAMES = ('闻韶宁', '闻绍宁')


class MultiVersionRouteTests(unittest.TestCase):
    def test_repeated_field_can_belong_to_different_near_names(self):
        question = '查闻韶宁的饮品首版，再查闻绍宁的饮品目前那份。'
        matches = mentions(question, NAMES, FIELDS)
        self.assertEqual([(item.owner, item.field) for item in matches],
                         [('闻韶宁', 'drink'), ('闻绍宁', 'drink')])
        whole, local = matches[1].views(question)
        self.assertIn('首版', whole)
        self.assertNotIn('首版', local)
        self.assertEqual(local, '再查闻绍宁的【饮品】目前那份')

    def test_inheritance_does_not_cross_unknown_subject_or_sentence(self):
        question = '查闻韶宁的饮品、见面地点。再说见面地点；查翁闻韶宁的饮品、见面地点。'
        self.assertEqual([item.owner for item in mentions(question, NAMES, FIELDS)],
                         ['闻韶宁', '闻韶宁', None, None, None])

    def test_each_field_resolves_its_own_version(self):
        drink, place = mentions('查闻韶宁的饮品和见面地点。', NAMES, FIELDS)
        days = [date(2026, 10, day) for day in (2, 6, 10)]
        catalog = {
            MemoryAddress('闻韶宁', 'drink'): [MemoryVersion(i, day, day) for i, day in enumerate(days, 1)],
            MemoryAddress('闻韶宁', 'place'): [MemoryVersion(4, days[0], days[0])],
        }
        self.assertEqual(resolve(catalog, drink, Lookup.FIRST, days[-1]), (1, 'located'))
        self.assertEqual(resolve(catalog, drink, Lookup.PREVIOUS, days[-1]), (2, 'located'))
        self.assertEqual(resolve(catalog, drink, Lookup.CURRENT, days[-1]), (3, 'located'))
        self.assertEqual(resolve(catalog, place, Lookup.PREVIOUS, days[-1]), (None, 'missing_version'))
        self.assertEqual(resolve(catalog, place, Lookup.SKIP, days[-1]), (None, 'not_requested'))

    def test_explicit_queries_preserve_uncertainty_instead_of_dropping_a_field(self):
        question = '查闻韶宁的饮品首版还是当前版；查闻绍宁的见面地点上一版。'
        candidates = mentions(question, NAMES, FIELDS)
        self.assertEqual(explicit_queries(question, candidates), [None, VersionQuery('previous')])

    def test_version_only_parses_locally_and_quoted_text_does_not_trigger_reading(self):
        question = '查闻韶宁的饮品第2版；见面地点暂不必查。'
        self.assertEqual(explicit_queries(question, mentions(question, NAMES, FIELDS)),
                         [VersionQuery('ordinal', 2), Lookup.SKIP])
        quoted = f'帮我改写「{question}」'
        self.assertEqual(explicit_queries(quoted, mentions(quoted, NAMES, FIELDS)), [Lookup.SKIP, Lookup.SKIP])

    def test_relative_version_does_not_fall_back_to_current(self):
        question = '查闻韶宁的饮品现行前一份；查闻绍宁的饮品当前版。'
        self.assertEqual(explicit_queries(question, mentions(question, NAMES, FIELDS)),
                         [VersionQuery('previous'), VersionQuery('current')])
        ambiguous = '查闻韶宁的饮品在现行记录以前用的那份。'
        self.assertEqual(explicit_queries(ambiguous, mentions(ambiguous, NAMES, FIELDS)), [None])
