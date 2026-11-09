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

    def test_covering_visits_every_window_before_repeating(self):
        tokenizer = HybridTokenizer(['甲', '乙', '丙'])
        docs = [{'split': 'train', 'sha256': 'one', 'text': '甲乙丙'}]
        windows, picks, state = reconstruct(docs, tokenizer, length=3, steps=2, batch=1,
                                            seed=9, sampling='covering-v1')
        self.assertEqual(len(windows), 2)
        self.assertEqual(len(set(picks)), 2)
        self.assertEqual(state['draws'], 2)
        self.assertEqual(set(state['counts'].values()), {1})


if __name__ == '__main__':
    unittest.main()
