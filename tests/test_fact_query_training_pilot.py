import unittest

from scripts.deepseek_fact_query_training_pilot_20261117 import cases, TRAIN_NAMES, TRAIN_FACTS, HELD_NAMES, HELD_FACTS


class FactTrainingCasesTests(unittest.TestCase):
    def test_counterbalanced_targets_and_disjoint_text(self):
        train, held = cases(TRAIN_NAMES, TRAIN_FACTS), cases(HELD_NAMES, HELD_FACTS)
        self.assertEqual(len(train), 96)
        self.assertEqual(len(held), 24)
        self.assertFalse({r['text'] for r in train} & {r['text'] for r in held})
        for group in (train, held):
            self.assertEqual(sum(r['kind'] == 'unknown' for r in group), len(group) // 3)
            for offset in range(0, len(group), 3):
                a, b, unknown = group[offset:offset + 3]
                self.assertEqual(a['text'].split('。')[0], b['text'].split('。')[0])
                self.assertNotEqual(a['text'][a['left']:a['right']], b['text'][b['left']:b['right']])
                self.assertIsNone(unknown['label'])
            for offset in range(0, len(group), 12):
                forward, reverse = group[offset], group[offset + 3]
                self.assertNotEqual(forward['text'].split('。')[0], reverse['text'].split('。')[0])
                self.assertEqual(forward['text'][forward['left']:forward['right']],
                                 reverse['text'][reverse['left']:reverse['right']])
