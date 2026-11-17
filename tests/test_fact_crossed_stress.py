import unittest
from scripts.deepseek_fact_crossed_stress_20261117 import cases


class FactStressCasesTests(unittest.TestCase):
    def test_known_and_absent_cases_and_homonyms(self):
        groups = cases()
        for name in ('paraphrase', 'same_type'):
            self.assertEqual(sum(r['kind'] == 'known' for r in groups[name]), 16)
            self.assertEqual(sum(r['kind'] == 'unknown' for r in groups[name]), 8)
            for row in groups[name]:
                if row['kind'] == 'known':
                    self.assertTrue(row['text'][row['left']:row['right']])
                else:
                    self.assertIsNone(row['label'])
        self.assertEqual(len(groups['homonym']), 8)
        self.assertTrue(all(row['label'] is None for row in groups['homonym']))
