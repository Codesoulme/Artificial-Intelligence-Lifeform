import unittest

import torch

from scripts.deepseek_spora_original_first_token_rank import describe


class Tokenizer:
    def decode(self, ids):
        return str(ids[0])


class OriginalFirstTokenRankTests(unittest.TestCase):
    def test_ranking_includes_full_vocabulary_and_ties(self):
        scores = torch.tensor([1., 3., 2., 3., 0., -1.])
        row = describe(scores, Tokenizer(), 2, 0)
        self.assertEqual(row['target_rank'], 3)
        self.assertEqual(row['margin_vs_best'], -1.)
        self.assertEqual(row['margin_vs_other'], 1.)
        self.assertEqual(len(row['top']), 5)

    def test_nonfinite_and_out_of_range_rejected(self):
        for scores, target, other in ((torch.tensor([1., float('nan')]), 0, 1),
                                      (torch.tensor([1., 2.]), 2, 0)):
            with self.assertRaises(ValueError):
                describe(scores, Tokenizer(), target, other)


if __name__ == '__main__':
    unittest.main()
