import unittest

from scripts.deepseek_spora_multirelation_train_pilot import checkpoint_due


class CheckpointScheduleTests(unittest.TestCase):
    def test_periodic_and_final_save_for_nonmultiple_run(self):
        self.assertEqual([step for step in range(1, 145) if checkpoint_due(step, 144)],
                         [32, 64, 96, 128, 144])

    def test_invalid_schedule_fails(self):
        for args in ((0, 144), (145, 144), (True, 144), (1, 2, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                checkpoint_due(*args)


if __name__ == '__main__':
    unittest.main()
