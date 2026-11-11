import unittest

from scripts.deepseek_spora_reader_full_answer_pilot import unique_token_span


class ReaderAnswerTokenSpanTests(unittest.TestCase):
    def test_locates_complete_multitoken_value(self):
        self.assertEqual(unique_token_span([8, 12, 34, 56, 9], [34, 56]), 2)
        self.assertEqual(unique_token_span([8, 12, 9], [12]), 1)

    def test_rejects_missing_ambiguous_or_empty_value(self):
        for sequence, value in (([1, 2], []), ([1, 2], [2, 3]), ([1, 1, 1], [1, 1])):
            with self.subTest(sequence=sequence, value=value), self.assertRaises(ValueError):
                unique_token_span(sequence, value)


if __name__ == '__main__':
    unittest.main()
