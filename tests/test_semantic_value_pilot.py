import unittest

from scripts.deepseek_semantic_value_pilot import HELD_TEMPLATES, TRAIN_TEMPLATES, WORDS, balanced_schedule, run, word_correct
from scripts.deepseek_semantic_value_diagnose import candidate_metrics
import torch


class SemanticValuePilotTests(unittest.TestCase):
    def test_word_scoring_requires_exact_answer(self):
        self.assertTrue(word_correct('咖啡。', '咖啡'))
        self.assertFalse(word_correct('咖啡或茶', '咖啡'))
        self.assertFalse(word_correct('水', '咖啡'))

    def test_vocabulary_and_prompt_split(self):
        self.assertEqual(len(set(WORDS)), 8)
        self.assertFalse(set(TRAIN_TEMPLATES) & set(HELD_TEMPLATES))
        self.assertTrue(all('{name}' in text for text in TRAIN_TEMPLATES + HELD_TEMPLATES))

    def test_candidate_ranking_distinguishes_full_vocabulary(self):
        logits = torch.tensor([[0., 2., 1., 3.], [0., 1., 2., 0.]])
        rows = candidate_metrics(logits, [1, 2], [0, 1])
        self.assertTrue(all(r['candidate_correct'] for r in rows))
        self.assertFalse(rows[0]['full_vocabulary_top_is_target'])
        self.assertTrue(rows[1]['full_vocabulary_top_is_target'])
        with self.assertRaises(ValueError):
            candidate_metrics(logits, [1, 1], [0, 1])

    def test_invalid_continuation_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            run(output='runs/semantic-test-invalid-schedule', steps=2, word_ratio=0)

    def test_balanced_schedule_rotates_all_values_and_pairs_distinct_states(self):
        schedule = balanced_schedule(32, 3, 17)
        self.assertEqual(schedule, balanced_schedule(32, 3, 17))
        for group in ('word', 'numeric_replay'):
            values = [row[4] for row in schedule if row[0] == group]
            self.assertEqual(len(set(values.count(value) for value in range(8))), 1)
        self.assertTrue(all(row[4] != row[5] for row in schedule))


if __name__ == '__main__':
    unittest.main()
