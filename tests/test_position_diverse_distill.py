import unittest

from scripts import deepseek_spora_position_diverse_distill as audit


class PositionDiverseDistillTests(unittest.TestCase):
    def test_extra_prompts_keep_conversation_groups_disjoint(self):
        train = [{'group': f'train-{index}'} for index in range(8)]
        held = [{'group': f'held-{index}'} for index in range(8)]
        selected = audit.extra_prompts(train, held, ['train-1', 'train-2', 'held-2', 'held-3'])
        self.assertEqual(len(selected), 6)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            audit.extra_prompts(train, held, ['train-0'])

    def test_only_long_training_positions_receive_updates(self):
        trajectories = [{'split': 'train', 'lengths': [96, 160, 224]},
                        {'split': 'held', 'lengths': [96, 160, 224]}]
        self.assertEqual(audit.training_positions(trajectories), [(0, 160), (0, 224)])


if __name__ == '__main__':
    unittest.main()
