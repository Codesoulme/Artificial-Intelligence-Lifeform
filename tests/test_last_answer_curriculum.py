import unittest

from scripts.prepare_last_answer_curriculum import final_reply_mask


class LastAnswerCurriculumTests(unittest.TestCase):
    def test_only_final_assistant_tokens_are_supervised(self):
        tokens = [0, 3, 10, 4, 20, 21, 3, 11, 4, 22, 23]
        mask = [False, False, False, False, True, True, False, False, False, True, True]
        result = final_reply_mask(tokens, mask, 4)
        self.assertEqual(result, [False] * 9 + [True, True])
        self.assertEqual(tokens[-2:], [22, 23])

    def test_rejects_missing_or_unaligned_final_reply(self):
        with self.assertRaises(ValueError):
            final_reply_mask([0, 4, 2], [False, False], 4)
        with self.assertRaises(ValueError):
            final_reply_mask([0, 4, 2], [False, False, False], 4)


if __name__ == "__main__":
    unittest.main()
