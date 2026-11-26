import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from spora.research_versioned_memory_pointer import VersionCoverageBias, VersionReplyActions


class VersionMemoryCoverageTests(unittest.TestCase):
    def test_ignores_prompt_and_adds_only_a_soft_continuation_bias(self):
        bias = VersionCoverageBias([2, 3, 4])
        scores = torch.zeros(1, 6)
        scores[0, 5] = 10.
        self.assertIs(bias(torch.tensor([[0, 2]]), scores), scores)
        shifted = bias(torch.tensor([[0, 2, 2]]), scores)
        self.assertEqual(shifted.argmax(-1).item(), 5)
        self.assertEqual(shifted[0, 3].item(), 8.)
        self.assertEqual(scores[0, 3].item(), 0.)

    def test_follows_an_overlapping_prefix_without_repeating_a_completed_value(self):
        bias = VersionCoverageBias([2, 2, 3])
        self.assertEqual(bias.next_token((2, 2)), 3)
        self.assertEqual(bias.next_token((2, 0, 2)), 2)
        self.assertIsNone(bias.next_token((2, 2, 3, 0, 2)))
        self.assertIsNone(bias.next_token((0, 1)))

    def test_empty_and_single_token_values_do_not_require_continuation(self):
        scores = torch.zeros(1, 6)
        for value in ([], [2]):
            with self.subTest(value=value):
                bias = VersionCoverageBias(value)
                self.assertIsNone(bias.next_token((2,)))
                self.assertIs(bias(torch.tensor([[0, 2]]), scores), scores)
                self.assertIsNone(bias.prompt_length)


class VersionReplyActionTests(unittest.TestCase):
    def test_shared_role_pointer_tracks_a_slot_swap_and_preserves_text_scores(self):
        head = VersionReplyActions(3, [4, 5], slots=2, field_count=4, shared_role_pointer=True)
        hidden = torch.tensor([[1., 2., 3.], [-1., 1., -2.]])
        fields = torch.tensor([[1, 2], [2, 3]])
        initial = head(hidden, torch.tensor([0, 1]), field_ids=fields)
        swapped = head(hidden, torch.tensor([0, 2]), field_ids=fields.flip(-1))
        torch.testing.assert_close(initial[:, :2], swapped[:, :2].flip(-1))
        torch.testing.assert_close(initial[:, 2:], swapped[:, 2:])
        self.assertTrue(bool(initial[1, 0].isneginf()))
        self.assertTrue(bool(swapped[1, 1].isneginf()))

    def test_role_ids_change_action_scores_and_preserve_copy_coverage(self):
        head = VersionReplyActions(3, [4], slots=2, field_count=3)
        with torch.no_grad():
            head.project.weight.zero_()
            head.project.bias.zero_()
            head.progress.weight.zero_()
            head.fields.weight.zero_()
            head.fields.weight[1, 0] = 3.
            head.fields.weight[2, 0] = -3.
            head.field_projection.weight.zero_()
            head.field_projection.weight[0, 0] = 1.
            head.actions.weight.zero_()
            head.actions.bias.zero_()
            head.actions.weight[0, 0] = 2.
        hidden, logits = torch.zeros(3), torch.zeros(6)
        options = dict(available=(True, True), memory_fraction=.95)
        correct = torch.tensor([1, 2])
        self.assertEqual(head.choose_slot(hidden, logits, copied=0, field_ids=correct, **options), (0, None))
        self.assertNotEqual(head.choose_slot(hidden, logits, copied=0, field_ids=correct.flip(0), **options), (0, None))
        self.assertNotEqual(head.choose_slot(hidden, logits, copied=1, field_ids=correct, **options), (0, None))
        with self.assertRaisesRegex(ValueError, 'Field IDs'):
            head(hidden, torch.tensor(0))

    def test_premature_eos_can_complete_pending_fields_without_masking_eos(self):
        from scripts.audit_spora_dual_version_reply import reply

        head = VersionReplyActions(3, [4], slots=2)
        with torch.no_grad():
            head.actions.weight.zero_()
            head.actions.bias.copy_(torch.tensor([-10., -9., 9.]))
        calls = []

        def model_forward(input_ids, **kwargs):
            calls.append(input_ids.tolist()[0])
            return SimpleNamespace(last_hidden_state=torch.zeros(1, 1, 3), past_key_values=object())

        model = SimpleNamespace(device='cpu', model=model_forward, lm_head=lambda hidden: torch.zeros(12))
        tokenizer = SimpleNamespace(eos_token_id=4, decode=lambda tokens, **_: str(tokens))
        values = (torch.tensor([1, 2]), torch.tensor([6, 7]))
        with patch('scripts.audit_spora_dual_version_reply.closed_think_inputs',
                   return_value={'input_ids': torch.tensor([[0]])}):
            output = reply(model, tokenizer, head, 'query', values, require_coverage=True,
                           complete_on_eos=True, max_new_tokens=5)
            short = reply(model, tokenizer, head, 'query', values, require_coverage=True,
                          complete_on_eos=True, max_new_tokens=2)
        self.assertEqual(output['token_ids'], [6, 7, 1, 2, 4])
        self.assertEqual(output['copied'], [{'slot': 1, 'at': 0}, {'slot': 0, 'at': 2}])
        self.assertEqual(calls[:3], [[0], [6, 7], [1, 2]])
        self.assertEqual(short['token_ids'], [4])
        self.assertEqual(short['copied'], [])

    def test_whole_value_action_competes_once_and_respects_the_remaining_budget(self):
        head = VersionReplyActions(3, [4])
        with torch.no_grad():
            head.actions.weight.zero_()
            head.actions.bias.copy_(torch.tensor([9., 0.]))
        hidden = torch.zeros(3)
        logits = torch.zeros(6)
        self.assertIsNone(head.choose(hidden, logits, copied=False, room_for_copy=True, memory_fraction=.95))
        self.assertEqual(head.choose(hidden, logits, copied=True, room_for_copy=True, memory_fraction=.95), 4)
        self.assertEqual(head.choose(hidden, logits, copied=False, room_for_copy=False, memory_fraction=.95), 4)

    def test_an_ordinary_token_can_win_before_a_copy_occurs(self):
        head = VersionReplyActions(3, [4])
        with torch.no_grad():
            head.actions.weight.zero_()
            head.actions.bias.copy_(torch.tensor([-9., 0.]))
        logits = torch.tensor([15., 0., 0., 0., 0., 0.])
        self.assertEqual(head.choose(torch.zeros(3), logits, copied=False, room_for_copy=True, memory_fraction=.2), 0)

    def test_two_whole_values_track_availability_and_completion_independently(self):
        head = VersionReplyActions(3, [4], slots=2)
        with torch.no_grad():
            head.actions.weight.zero_()
            head.actions.bias.copy_(torch.tensor([8., 9., 0.]))
        hidden, logits = torch.zeros(3), torch.zeros(6)
        for copied, available, expected in (
            (0, (True, True), (1, None)),
            (0, (True, False), (0, None)),
            (1, (True, True), (1, None)),
            (2, (True, True), (0, None)),
            (3, (True, True), (None, 4)),
            (0, (False, False), (None, 4)),
        ):
            with self.subTest(copied=copied, available=available):
                self.assertEqual(head.choose_slot(hidden, logits, copied=copied, available=available,
                                                  memory_fraction=.95), expected)

    def test_blocked_end_token_is_removed_from_both_distributions(self):
        head = VersionReplyActions(3, [4], slots=2)
        with torch.no_grad():
            head.actions.weight.zero_()
            head.actions.bias.copy_(torch.tensor([-9., -9., 9.]))
        hidden = torch.zeros(3)
        logits = torch.tensor([0., 10., 0., 0., 20., 0.])
        options = dict(copied=1, available=(True, True), memory_fraction=.2)
        self.assertEqual(head.choose_slot(hidden, logits, **options), (None, 4))
        self.assertEqual(head.choose_slot(hidden, logits, blocked_token=4, **options), (None, 1))
        self.assertEqual(logits[4].item(), 20.)
