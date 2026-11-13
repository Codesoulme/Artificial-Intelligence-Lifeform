import unittest

from scripts.deepseek_spora_multirelation_train_pilot import checkpoint_due, compare_recovery_rows


class CheckpointScheduleTests(unittest.TestCase):
    def test_periodic_and_final_save_for_nonmultiple_run(self):
        self.assertEqual([step for step in range(1, 145) if checkpoint_due(step, 144)],
                         [32, 64, 96, 128, 144])

    def test_invalid_schedule_fails(self):
        for args in ((0, 144), (145, 144), (True, 144), (1, 2, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                checkpoint_due(*args)

    def test_exact_recovery_matches_by_case_not_order(self):
        first = {'group': 'x', 'person': 'a', 'domain': 'd', 'fact': 'f',
                 'question': 'q', 'token_ids': [1]}
        second = {**first, 'fact': 'g', 'token_ids': [2]}
        result = compare_recovery_rows([first, second], [second, first], 144, 144)
        self.assertTrue(result['exact_replay'])
        with self.assertRaisesRegex(ValueError, 'did not reproduce'):
            compare_recovery_rows([first], [first, second], 144, 144)
        different_step = compare_recovery_rows([first], [first, second], 128, 144)
        self.assertFalse(different_step['exact_replay'])
        self.assertEqual(different_step['comparable_cases'], 1)


if __name__ == '__main__':
    unittest.main()
