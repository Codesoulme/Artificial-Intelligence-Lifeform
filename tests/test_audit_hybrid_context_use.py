import unittest

from scripts.audit_hybrid_context_use import samples, shuffled_prefix
from spora.hybrid_tokenizer import HybridTokenizer


class ContextAuditTests(unittest.TestCase):
    def test_shuffle_preserves_boundaries_and_multiset(self):
        prefix = [256, 1, 2, 3, 4, 5, 6]
        shuffled = shuffled_prefix(prefix, 42)
        self.assertEqual(shuffled, shuffled_prefix(prefix, 42))
        self.assertEqual(shuffled[0], prefix[0])
        self.assertEqual(shuffled[-1], prefix[-1])
        self.assertCountEqual(shuffled, prefix)
        with self.assertRaises(ValueError):
            shuffled_prefix([256, 1], 1)

    def test_only_development_targets_and_no_eos(self):
        tokenizer = HybridTokenizer(['甲', '乙', '丙', '丁'])
        docs = [{'split': 'train', 'text': '甲乙丙丁', 'sha256': 'a', 'path': 'train'},
                {'split': 'development', 'text': '甲乙丙丁', 'sha256': 'b', 'path': 'dev'}]
        rows = samples(docs, tokenizer, (3, 5))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['path'], 'dev')
        self.assertEqual(rows[0]['target'], tokenizer.ids['丙'])


if __name__ == '__main__':
    unittest.main()
