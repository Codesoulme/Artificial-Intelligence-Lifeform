import unittest

from scripts.deepseek_semantic_value_pilot import HELD_TEMPLATES, TRAIN_TEMPLATES, WORDS, word_correct


class SemanticValuePilotTests(unittest.TestCase):
    def test_word_scoring_requires_exact_answer(self):
        self.assertTrue(word_correct('咖啡。', '咖啡'))
        self.assertFalse(word_correct('咖啡或茶', '咖啡'))
        self.assertFalse(word_correct('水', '咖啡'))

    def test_vocabulary_and_prompt_split(self):
        self.assertEqual(len(set(WORDS)), 8)
        self.assertFalse(set(TRAIN_TEMPLATES) & set(HELD_TEMPLATES))
        self.assertTrue(all('{name}' in text for text in TRAIN_TEMPLATES + HELD_TEMPLATES))


if __name__ == '__main__':
    unittest.main()
