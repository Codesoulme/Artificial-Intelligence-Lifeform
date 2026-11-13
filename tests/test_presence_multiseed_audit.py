import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.spora_presence_multiseed_reload_audit import measure


class PresenceAuditTests(unittest.TestCase):
    def test_presence_and_wrong_value_are_reported_separately(self):
        model = SimpleNamespace(config=SimpleNamespace(key_count=4, value_count=3, unknown=2),
                                parameters=lambda: iter([torch.zeros(1)]))
        logits = torch.tensor([[5., 0., 0.], [0., 5., 0.], [0., 0., 5.], [0., 5., 0.]])
        with patch('scripts.spora_presence_multiseed_reload_audit.query', return_value=logits):
            summary = measure(model, object(), {0: 0, 1: 0, 2: 1})['summary']
        self.assertEqual(summary['false_known'], 1)
        self.assertEqual(summary['false_unknown'], 1)
        self.assertEqual(summary['known_value_correct'], 1)
        self.assertEqual(summary['wrong_known_value'], 1)


if __name__ == '__main__':
    unittest.main()
