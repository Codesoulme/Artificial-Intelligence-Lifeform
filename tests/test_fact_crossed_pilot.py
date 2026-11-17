import unittest
from collections import Counter

from scripts.deepseek_fact_crossed_pilot_20261117 import cases, NAMES_TRAIN, FACTS_TRAIN


class CrossedFactCasesTests(unittest.TestCase):
    def test_every_fact_can_be_present_or_absent(self):
        rows = cases(NAMES_TRAIN, FACTS_TRAIN)
        self.assertEqual(len(rows), 72)
        self.assertEqual(len({r['text'] for r in rows}), len(rows))
        counts = Counter((r['fact'], r['kind']) for r in rows)
        for fact in FACTS_TRAIN:
            self.assertEqual(counts[fact, 'known'], 16)
            self.assertEqual(counts[fact, 'unknown'], 8)
        for offset in range(0, len(rows), 3):
            first, second, unknown = rows[offset:offset + 3]
            self.assertNotEqual(first['text'][first['left']:first['right']],
                                second['text'][second['left']:second['right']])
            self.assertIsNone(unknown['label'])
