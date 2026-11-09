import unittest

import torch

from scripts.deepseek_observation_value_probe import ABSTAIN, WORDS, cases, metrics


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


if __name__ == '__main__':
    unittest.main()
