import unittest

import torch

from scripts.deepseek_observation_value_probe import ABSTAIN, WORDS, cases, metrics
from scripts.deepseek_observation_value_probe import EXTRA_NEGATIVE, EXTRA_POSITIVE, TRAIN_NAMES, TRAIN_POSITIVE, TRAIN_NEGATIVE
from scripts.deepseek_observation_value_audit import NEXT_NAMES, NEXT_POSITIVE, NEXT_NEGATIVE


class ObservationValueProbeTests(unittest.TestCase):
    def test_predictions_count_false_writes_and_correct_values_separately(self):
        rows = [{'text': 'known', 'label': 0, 'group': 'positive'},
                {'text': 'unknown', 'label': ABSTAIN, 'group': 'unsupported'}]
        logits = torch.full((2, ABSTAIN + 1), -10.)
        logits[0, 0] = 10.
        logits[1, 1] = 10.
        result = metrics(logits, rows)
        self.assertEqual(result['summary']['positive']['correct'], 1)
        self.assertEqual(result['summary']['unsupported']['correct'], 0)
        self.assertEqual(result['summary']['unsupported']['would_write'], 1)

    def test_contradictory_cases_get_abstain_labels(self):
        rows = cases(('人名',), ('{name}喜欢{word}。',), {'negative': ('{name}不喜欢{word}。',)})
        self.assertEqual(len(rows), 2 * len(WORDS))
        self.assertTrue(all(row['label'] == ABSTAIN for row in rows if row['group'] == 'negative'))

    def test_independent_audit_uses_distinct_names_and_text(self):
        training = cases(TRAIN_NAMES, TRAIN_POSITIVE, TRAIN_NEGATIVE) + cases(TRAIN_NAMES, EXTRA_POSITIVE, EXTRA_NEGATIVE)
        evaluation = cases(NEXT_NAMES, NEXT_POSITIVE, NEXT_NEGATIVE)
        self.assertFalse(set(TRAIN_NAMES) & set(NEXT_NAMES))
        self.assertFalse({row['text'] for row in training} & {row['text'] for row in evaluation})


if __name__ == '__main__':
    unittest.main()
