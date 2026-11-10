import unittest

from scripts.deepseek_temporal_intent_router import CURRENT, HELD, LATER, PAST, TRAIN, examples, route_spec


class TemporalIntentRouterTests(unittest.TestCase):
    def test_training_and_holdout_are_balanced_disjoint_phrases(self):
        train, held = examples(TRAIN), examples(HELD)
        self.assertEqual(len(train), 12)
        self.assertEqual(len(held), 9)
        self.assertEqual({row['label'] for row in train}, {CURRENT, PAST, LATER})
        self.assertEqual({row['label'] for row in held}, {CURRENT, PAST, LATER})
        self.assertFalse({row['text'] for row in train} & {row['text'] for row in held})

    def test_predicted_intent_selects_its_own_memory_time(self):
        self.assertEqual(route_spec(CURRENT), (0, None))
        self.assertEqual(route_spec(PAST), (1, 0))
        self.assertEqual(route_spec(LATER), (1, 1))
        with self.assertRaises(ValueError):
            route_spec(99)


if __name__ == '__main__':
    unittest.main()
