import unittest

import torch
from scripts.deepseek_spora_writer_norm_audit import normalize_to_training_norm


class WriterNormAuditTests(unittest.TestCase):
    def test_norm_and_direction(self):
        value = torch.tensor([3., 4.])
        result = normalize_to_training_norm(value, 2.)
        torch.testing.assert_close(result, torch.tensor([1.2, 1.6]))

    def test_rejects_invalid_vectors_or_training_norm(self):
        for value, norm in ((torch.zeros(2), 1.), (torch.tensor([float('nan')]), 1.),
                            (torch.ones(2), 0.), (torch.ones(2), float('inf'))):
            with self.subTest(value=value, norm=norm), self.assertRaises(ValueError):
                normalize_to_training_norm(value, norm)


if __name__ == '__main__':
    unittest.main()
