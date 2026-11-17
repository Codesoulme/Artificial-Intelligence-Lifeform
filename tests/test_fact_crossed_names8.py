import unittest
from scripts.deepseek_fact_crossed_names8_20261117 import NAMES
from scripts.deepseek_fact_crossed_pilot_20261117 import NAMES_HELD, FACTS_TRAIN, cases


class ExpandedNameCasesTests(unittest.TestCase):
    def test_names_unique_and_excluded_from_held_group(self):
        names = [name for pair in NAMES for name in pair]
        self.assertEqual(len(names), 16)
        self.assertEqual(len(set(names)), 16)
        self.assertFalse(set(names) & {name for pair in NAMES_HELD for name in pair})
        self.assertEqual(len(cases(NAMES, FACTS_TRAIN)), 288)
