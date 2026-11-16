import unittest
from unittest.mock import patch
import random

import torch

from scripts.deepseek_spora_ttt_shared_joint_pilot import (make_query_batch, exact, evaluate, sample_episode,
                                                          write_sequence, paired_margin_loss, change_last_event)
from spora.research_ttt_ntp import sequential_ntp_write


class SharedJointTests(unittest.TestCase):
    def test_paired_margin_penalizes_unchanged_predictions(self):
        same = paired_margin_loss(torch.tensor([2., 0.]), torch.tensor([2., 0.]), 0, 1)
        switched = paired_margin_loss(torch.tensor([2., 0.]), torch.tensor([0., 2.]), 0, 1)
        self.assertLess(switched, same)
        with self.assertRaises(ValueError):
            paired_margin_loss(torch.ones(2), torch.ones(2), 0, 0)

    def test_pair_changes_only_last_write_of_requested_person(self):
        original = [('A', 'old'), ('B', 'other'), ('A', 'new')]
        changed = change_last_event(original, 'A', 'next')
        self.assertEqual(changed, [('A', 'old'), ('B', 'other'), ('A', 'next')])
        self.assertEqual(original[-1], ('A', 'new'))
        with self.assertRaises(ValueError):
            change_last_event(original, 'missing', 'x')

    def test_multi_token_write_matches_sequential_reference_and_keeps_gradients(self):
        wrapper = type('Wrapper', (), {'base': torch.nn.Linear(3, 2, bias=False)})()
        wrappers = {0: wrapper}
        keys = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        targets = torch.tensor([[2., 3.], [4., 5.]], requires_grad=True)
        cache = {'keys': {0: keys}, 'targets': {0: targets}}
        actual = write_sequence(wrappers, [cache], None, differentiable=True)
        reference = {0: torch.zeros(2, 3)}
        sequential_ntp_write(reference, {0: keys}, {0: targets.detach()}, eta=.5, ridge=1.)
        torch.testing.assert_close(actual[0], reference[0])
        actual[0].sum().backward()
        self.assertTrue(torch.all(targets.grad.abs().sum(dim=1) > 0))

    def test_exact_strips_one_terminal_punctuation(self):
        self.assertTrue(exact('豆浆。', '豆浆'))
        self.assertFalse(exact('酸奶', '豆浆'))

    def test_query_batch_keeps_each_prefix_and_masks_padding(self):
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 9

            def encode(self, value, add_special_tokens=False):
                return [7]

        # Avoid loading the real model in this shape test.
        import scripts.deepseek_spora_ttt_shared_joint_pilot as pilot_module
        original = pilot_module.pilot.closed_think_inputs
        try:
            pilot_module.pilot.closed_think_inputs = lambda tokenizer, question, device: {
                'input_ids': torch.tensor([[1, 2]], dtype=torch.long),
            }
            batch = make_query_batch(Tokenizer(), [('小林', '豆浆', '问{name}'),
                                                   ('阿宁', '咖啡', '问{name}')])
        finally:
            pilot_module.pilot.closed_think_inputs = original
        self.assertEqual(tuple(batch['input_ids'].shape), (2, 4))
        self.assertTrue(torch.equal(batch['attention_mask'], torch.ones(2, 4, dtype=torch.long)))
        self.assertEqual([row[0].numel() for row in batch['rows']], [2, 2])

    def test_correction_evaluation_uses_original_then_corrected_event(self):
        caches = {('A', 'old'): 'old-cache', ('A', 'new'): 'new-cache'}
        events = [('A', 'old'), ('A', 'new')]
        cases = [('A', 'new', 'ask {name}')]
        with patch('scripts.deepseek_spora_ttt_shared_joint_pilot.write_sequence',
                   return_value='memory') as writer, patch(
                       'scripts.deepseek_spora_ttt_shared_joint_pilot.pilot.generate',
                       return_value={'answer': 'new', 'token_ids': [1]}):
            rows = evaluate(None, None, None, None, caches, cases, events=events)
        self.assertEqual(writer.call_args.args[1], ['old-cache', 'new-cache'])
        self.assertTrue(rows[0]['exact'])

    def test_random_bindings_and_last_event_define_expected(self):
        rng = random.Random(31)
        assignments = set()
        for step in range(64):
            events, expected, correction = sample_episode(rng, step)
            latest = dict(events)
            self.assertEqual(latest, dict(expected))
            self.assertEqual(len(events), 5 if correction else 4)
            assignments.add(tuple(expected))
            if correction:
                person, updated = events[-1]
                self.assertNotEqual(dict(events[:-1])[person], updated)
        self.assertGreater(len(assignments), 10)


if __name__ == '__main__':
    unittest.main()
