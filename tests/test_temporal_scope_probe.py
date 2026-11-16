import unittest

import torch

from scripts import deepseek_spora_temporal_scope_probe_20261117 as probe


class TemporalScopeProbeTests(unittest.TestCase):
    def test_splits_preserve_all_classes_and_do_not_share_text(self):
        splits = {name: probe.build(name) for name in ('train', 'cal', 'test')}
        self.assertEqual({name: len(rows) for name, rows in splits.items()},
                         {'train': 48, 'cal': 16, 'test': 24})
        texts = {name: {row['question'] for row in rows} for name, rows in splits.items()}
        for left, right in (('train', 'cal'), ('train', 'test'), ('cal', 'test')):
            self.assertFalse(texts[left] & texts[right])
        for rows in splits.values():
            self.assertEqual({row['label'] for row in rows}, {0, 1, 2, 3})

    def test_metrics_distinguish_wrong_read_from_abstention(self):
        result = probe.metrics(torch.tensor([1, 3, 0, 3]), torch.tensor([1, 0, 3, 3]))
        self.assertEqual(result, {'cases': 4, 'correct': 2, 'wrong_concrete': 1, 'false_abstain': 1})

    def test_prediction_requires_confidence_and_explicit_abstention(self):
        logits = torch.tensor([[8., 0., 0., 0.], [0., 0., 0., 8.], [0., 0., 0., 0.]])
        self.assertEqual(probe.predict(logits, .7).tolist(), [0, 3, 3])


if __name__ == '__main__':
    unittest.main()
