import unittest

from scripts.deepseek_fact_query_attribution_audit_20261117 import NAMES, make_cases


class FactQueryAttributionTests(unittest.TestCase):
    def test_queries_change_target_without_changing_evidence(self):
        rows = make_cases()
        self.assertEqual(len(rows), len(NAMES) * 6)
        self.assertEqual(len({row['text'] for row in rows}), len(rows))
        for offset in range(0, len(rows), 3):
            drink, color, unknown = rows[offset:offset + 3]
            self.assertEqual(drink['text'].split('。')[0], color['text'].split('。')[0])
            self.assertEqual(color['text'].split('。')[0], unknown['text'].split('。')[0])
            self.assertNotEqual(drink['text'][drink['left']:drink['right']],
                                color['text'][color['left']:color['right']])
            self.assertIsNone(unknown['label'])
            self.assertIsNone(unknown['left'])
        for offset in range(0, len(rows), 6):
            first, swapped = rows[offset], rows[offset + 3]
            self.assertNotEqual(first['text'][first['left']:first['right']],
                                swapped['text'][swapped['left']:swapped['right']])
