import unittest

import torch

from scripts.audit_hybrid_sampling import reconstruct, summarize
from spora.hybrid_tokenizer import HybridTokenizer


class SamplingAuditTests(unittest.TestCase):
    def test_reconstruct_matches_torch_sampler_and_counts_duplicate_draws(self):
        tokenizer = HybridTokenizer(['甲', '乙', '丙'])
        docs = [{'split': 'train', 'sha256': 'one', 'text': '甲乙丙'},
                {'split': 'development', 'sha256': 'two', 'text': '甲乙'}]
        windows, picks, rng = reconstruct(docs, tokenizer, length=3, steps=4, batch=2, seed=9)
        self.assertEqual(windows, [('one', 2), ('one', 2)])
        generator = torch.Generator().manual_seed(9)
        expected = []
        for _ in range(4):
            expected.extend(torch.randint(2, (2,), generator=generator).tolist())
        self.assertEqual(picks, expected)
        self.assertTrue(torch.equal(rng, generator.get_state()))
        summary = summarize(windows, picks, {'one'})
        self.assertEqual(summary['target_presentations'], 16)
        self.assertEqual(summary['different_window_targets'], 2 * len(set(picks)))
        self.assertEqual(summary['documents_in_previous_pool'], 1)
        self.assertEqual(summary['documents_not_in_previous_pool'], 0)


if __name__ == '__main__':
    unittest.main()
