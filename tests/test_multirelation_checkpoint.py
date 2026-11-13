import unittest

import torch
from types import SimpleNamespace
from torch import nn
from scripts.deepseek_spora_multirelation_train_pilot import (checkpoint_due, compare_recovery_rows,
                                                              choose_counterfactual, counterfactual_loss,
                                                              optimizer_groups, validate_domains,
                                                              aligned_teacher_kl,
                                                              add_old_value_replay)


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

    def test_counterfactual_schedule_and_margin_direction(self):
        values = ('a', 'b', 'c')
        self.assertEqual({choose_counterfactual(values, 'a', cycle) for cycle in range(4)}, {'b', 'c'})
        with self.assertRaises(ValueError):
            choose_counterfactual(('a', 'a'), 'a', 0)
        own, swapped = torch.tensor(1., requires_grad=True), torch.tensor(2., requires_grad=True)
        counterfactual_loss(own, swapped).backward()
        self.assertGreater(float(own.grad), 0.)
        self.assertLess(float(swapped.grad), 0.)

    def test_memory_learning_rates_preserve_separate_original_group(self):
        module = SimpleNamespace(memory_key=nn.Linear(2, 2), memory_value=nn.Linear(2, 2),
                                 gain=nn.Parameter(torch.tensor(0.)))
        base = nn.Parameter(torch.ones(1))
        params = [base, *module.memory_key.parameters(), *module.memory_value.parameters(), module.gain]
        groups = optimizer_groups(params, module, 0.0001, 0.001)
        self.assertEqual(groups[0]['params'], [base])
        self.assertEqual(groups[1]['lr'], 0.0001)
        self.assertIs(groups[2]['params'][0], module.gain)
        with self.assertRaises(ValueError):
            optimizer_groups(params, module, -1, 0.001)

    def test_training_and_held_values_must_not_overlap(self):
        valid = {'x': ('r', ('{name} one', '{name} two'), ('a', 'b'), ('c',))}
        validate_domains(valid)
        invalid = {'x': ('r', ('{name} one', '{name} two'), ('a', 'b'), ('b',))}
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            validate_domains(invalid)

    def test_replay_only_repeats_older_supervised_values(self):
        domains = {'x': ('r', ('{name} one', '{name} two'), ('a', 'b', 'c'), ('held',))}
        cases = [{'domain': 'x', 'fact': v} for v in ('a', 'b', 'c')]
        replay = add_old_value_replay(cases, domains, old_count=2, factor=3)
        self.assertEqual([r['fact'] for r in replay].count('a'), 3)
        self.assertEqual([r['fact'] for r in replay].count('b'), 3)
        self.assertEqual([r['fact'] for r in replay].count('c'), 1)
        with self.assertRaisesRegex(ValueError, 'supervised'):
            add_old_value_replay([{'domain': 'x', 'fact': 'held'}], domains, 2, 2)

    def test_teacher_kl_validates_shapes_and_backpropagates_student_only(self):
        student = torch.randn(2, 5, requires_grad=True)
        teacher = torch.randn(2, 5, requires_grad=True)
        loss = aligned_teacher_kl(student, teacher, 2.0)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)
        with self.assertRaises(ValueError):
            aligned_teacher_kl(student, torch.randn(1, 5), 2.0)


if __name__ == '__main__':
    unittest.main()
