import unittest

import torch

from scripts.audit_spora_dense_joint_address import NONE
from scripts.audit_spora_switch_address_ranking import apply_split

from scripts.audit_spora_recurrent_lookup_scope import ASK, READ, ScopePlanner, decisions, episodes
from scripts.audit_spora_lookup_transitions import (
    ADD, KEEP, REMOVE, HELD_PAIR, LookupTransitions, apply_operations, normalize_row, operation_targets,
    paired_rows, paired_scope_metrics, transition_features,
)


class RecurrentLookupScopeTests(unittest.TestCase):
    def test_split_proposal_rejects_missing_target_without_substitution(self):
        scores = [(1, .93, .92), (1, .93, .92), (1, .93, .2), (1, .2, .92)]
        masks = torch.ones(4, NONE, dtype=torch.bool)
        masks[1, 1] = False
        self.assertEqual(apply_split(scores, masks, (.65, .65)), [(1, 1.), (NONE, 1.), (NONE, 1.), (NONE, 1.)])

    def test_candidate_swap_changes_positions_but_preserves_state_and_version(self):
        planner = ScopePlanner(8, size=6)
        query, events = torch.randn(3, 2, 8), torch.randn(3, 2, 2, 8)
        valid = torch.tensor([[True, True], [True, False], [False, False]])
        state = planner.advance(events, valid, recurrent=True)
        swapped_state = planner.advance(events.flip(2), valid, recurrent=True)
        torch.testing.assert_close(swapped_state, state.flip(1))
        self.assertEqual(torch.count_nonzero(state[-1]).item(), 0)
        actions, versions = planner.plan(query, state)
        reversed_actions, reversed_versions = planner.plan(query.flip(1), swapped_state)
        torch.testing.assert_close(reversed_actions[:, :READ], actions[:, :READ])
        torch.testing.assert_close(reversed_actions[:, READ:], actions[:, READ:].flip(1))
        torch.testing.assert_close(reversed_versions, versions)

    def test_held_prompts_pair_identical_questions_with_distinct_expected_actions(self):
        held = episodes('held')
        group = [row for row in held if row.addresses == held[0].addresses and row.question == '首版。']
        by_state = {row.scope: row.action for row in group}
        self.assertEqual(by_state['retain'], READ)
        self.assertEqual(by_state['replace'], READ + 1)
        self.assertEqual(by_state['both'], by_state['cancel'])
        self.assertLess(by_state['cancel'], READ)

    def test_read_requires_exactly_one_active_candidate_matching_the_plan(self):
        route = torch.tensor([[0., 0., 8., 0.]]).repeat(4, 1)
        versions = torch.zeros(4, 3)
        members = torch.tensor([[True, True], [False, False], [False, True], [True, False]])
        self.assertEqual(decisions((route, versions), .5, members), [(ASK, 0)] * 3 + [(READ, 0)])

    def test_annotated_turns_update_only_the_requested_candidates(self):
        rows = [next(row for row in episodes('held') if row.scope == kind)
                for kind in ('replace', 'both', 'cancel', 'mention')]
        opening, followup = operation_targets(rows).unbind(1)
        active = apply_operations(torch.zeros(4, 2, dtype=torch.bool), opening)
        self.assertEqual(active.tolist(), [[True, False]] * 4)
        self.assertEqual(followup.tolist(), [[REMOVE, ADD], [KEEP, ADD], [REMOVE, REMOVE], [KEEP, KEEP]])
        self.assertEqual(apply_operations(active, followup).tolist(),
                         [[False, True], [True, True], [False, False], [True, False]])

    def test_relative_markers_follow_the_named_address_when_swapped(self):
        row = next(row for row in episodes('held') if row.scope == 'replace')
        original, alternative = normalize_row(row)
        self.assertIn('本项记录', original.events[0])
        self.assertIn('另一项记录', original.events[1])
        self.assertIn('另一项记录', alternative.events[0])
        self.assertIn('本项记录', alternative.events[1])

    def test_confirmed_opening_leaves_absent_episodes_empty(self):
        head = LookupTransitions(3)
        head.operation = torch.nn.Identity()
        history = torch.tensor([[[[0., 1., 0.], [1., 0., 0.]], [[0., 0., 1.], [0., 1., 0.]]]]).repeat(2, 1, 1, 1)
        valid = torch.tensor([[True, True], [False, False]])
        confirmed = torch.tensor([[True, False], [True, False]])
        self.assertEqual(head.advance(history, valid, opening=confirmed).tolist(),
                         [[[True, False], [False, True]], [[False, False], [False, False]]])

    def test_paired_revisions_match_opening_and_score_both_outcomes(self):
        row = next(row for row in episodes('held') if row.scope == 'replace' and row.version >= 0)
        pairs = paired_rows([row], (HELD_PAIR,))
        (append, switch), = pairs
        self.assertEqual((append.events[0], append.question, append.addresses),
                         (switch.events[0], switch.question, switch.addresses))
        self.assertEqual((append.action, switch.action), (ASK, READ + 1))
        opening, followup = operation_targets([append, switch]).unbind(1)
        initial = apply_operations(torch.zeros(2, 2, dtype=torch.bool), opening)
        final = apply_operations(initial, followup)
        snapshots = torch.stack((initial, final), dim=1)
        self.assertEqual(paired_scope_metrics(snapshots, pairs), 1)
        self.assertEqual(paired_scope_metrics(snapshots.flip(0), pairs), 0)

    def test_clause_features_leave_version_input_and_missing_history_unchanged(self):
        cases = episodes('held')
        rows = [next(row for row in cases if row.scope == kind) for kind in ('absent', 'replace')]

        def encode(texts):
            return [torch.full((len(text), 1), float(len(text))) for text in texts]

        original = transition_features(encode, {'held': rows}, utterance_features=True)['held']
        multi = transition_features(encode, {'held': rows}, utterance_features=True, clause_features=True)['held']
        torch.testing.assert_close(original[0], multi[0])
        torch.testing.assert_close(original[2], multi[2])
        torch.testing.assert_close(original[1], multi[1][..., :1])
        self.assertFalse(multi[1][0].any().item())
        self.assertTrue((multi[1][1, 1, :, 1] < multi[1][1, 1, :, 0]).all().item())
