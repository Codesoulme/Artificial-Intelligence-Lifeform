import unittest

from scripts.deepseek_temporal_intent_router import CURRENT, HELD, LATER, PAST, TRAIN, examples


class TemporalIntentRouterTests(unittest.TestCase):
    def test_training_and_holdout_are_balanced_disjoint_phrases(self):
        train, held = examples(TRAIN), examples(HELD)
        self.assertEqual(len(train), 12)
        self.assertEqual(len(held), 9)
        self.assertEqual({row['label'] for row in train}, {CURRENT, PAST, LATER})
        self.assertEqual({row['label'] for row in held}, {CURRENT, PAST, LATER})
        self.assertFalse({row['text'] for row in train} & {row['text'] for row in held})


if __name__ == '__main__':
    unittest.main()
