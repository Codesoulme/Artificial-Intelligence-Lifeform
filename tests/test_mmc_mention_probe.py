import unittest

from scripts.deepseek_local_baseline import MODEL
from scripts.deepseek_mmc_mention_probe import AutoTokenizer, mention_masks


@unittest.skipUnless(MODEL.exists(), 'Local DeepSeek tokenizer is unavailable')
class MentionMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        cls.tokenizer.padding_side = 'right'

    def test_masks_select_distinct_mention_tokens_in_padded_batch(self):
        texts = ['1:<m1>\u6211</m1>\u559c\u6b22\u8336\n2:<m2>\u4f60</m2>',
                 '1:<m1>\u4f60</m1>\u5462\n1:<m2>\u6211</m2>']
        first, second = mention_masks(self.tokenizer, texts)
        self.assertEqual(first.sum(-1).tolist(), [1, 1])
        self.assertEqual(second.sum(-1).tolist(), [1, 1])
        self.assertFalse(bool((first & second).any()))

    def test_rejects_duplicate_or_reversed_markers(self):
        for text in ('<m1>\u6211</m1><m1>\u6211</m1><m2>\u4f60</m2>',
                     '<m2>\u4f60</m2><m1>\u6211</m1>', '<m1></m1><m2>\u4f60</m2>'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                mention_masks(self.tokenizer, [text])


if __name__ == '__main__':
    unittest.main()
