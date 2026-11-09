from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from spora.current_memory_status import current_memory_status


class CurrentMemoryStatusTests(unittest.TestCase):
    def setUp(self):
        self.model = nn.Linear(1, 1)
        self.model.config = SimpleNamespace(key_count=3, value_count=3, unknown=2)

    def test_status_does_not_return_the_memory_value(self):
        with patch('spora.current_memory_status.query', return_value=torch.tensor([[4., 0., -1.]])):
            result = current_memory_status(self.model, object(), 1)
        self.assertEqual(result['status'], 'known')
        self.assertEqual(set(result), {'status', 'unknown_probability'})
        with patch('spora.current_memory_status.query', return_value=torch.tensor([[0., 1., 4.]])):
            self.assertEqual(current_memory_status(self.model, object(), 1)['status'], 'unknown')

    def test_invalid_key_and_nonfinite_logits_are_rejected(self):
        for key in (-1, 3, True):
            with self.assertRaises(ValueError):
                current_memory_status(self.model, object(), key)
        with patch('spora.current_memory_status.query', return_value=torch.tensor([[0., float('nan'), 1.]])):
            with self.assertRaises(ValueError):
                current_memory_status(self.model, object(), 1)
