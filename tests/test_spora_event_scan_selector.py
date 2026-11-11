import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.spora_event_scan_selector import select_event


class EventScanTests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(config=SimpleNamespace(key_count=8))
        self.state = SimpleNamespace(cursor=6, internal=SimpleNamespace(start_cursor=0))
        self.rows = [{'action': 'new'}, {'action': 'repeat'}, {'action': 'correct'},
                     {'reason': 'unknown_event', 'source': None}, {'action': 'defer'}, {'action': 'correct'}]

    def test_order_selects_belief_changes_only(self):
        with patch('scripts.spora_event_scan_selector.read_provenance_batch', return_value=self.rows):
            for order, expected in (('first', 0), ('previous', 2), ('latest', 5)):
                result = select_event(self.model, self.state, 1, order)
                self.assertEqual(result['event_time'], expected)
                self.assertEqual(result['status'], 'experimental_candidate')
                self.assertIn('neural_unknown', result['warning'])

    def test_uncertainty_incomplete_history_and_budget_abstain(self):
        with patch('scripts.spora_event_scan_selector.read_provenance_batch',
                   return_value=[{'action': 'new'}, {'reason': 'uncertain_numeric_provenance'}]):
            self.assertEqual(select_event(self.model, self.state, 1, 'latest')['reason'], 'uncertain_provenance')
        self.assertEqual(select_event(self.model, self.state, 1, 'latest', max_scan=5)['reason'], 'scan_budget_exceeded')
        self.state.internal.start_cursor = 1
        self.assertEqual(select_event(self.model, self.state, 1, 'latest')['reason'], 'incomplete_provenance_history')

    def test_single_candidate_has_no_previous(self):
        with patch('scripts.spora_event_scan_selector.read_provenance_batch', return_value=[{'action': 'new'}]):
            self.assertEqual(select_event(self.model, self.state, 1, 'previous')['reason'], 'insufficient_observed_candidates')

    def test_effective_time_uses_latest_observation_at_each_effective_time(self):
        rows = [{'action': 'new', 'valid_at': 0}, {'action': 'correct', 'valid_at': 2},
                {'action': 'repeat', 'valid_at': 2}, {'action': 'correct', 'valid_at': 5}]
        with patch('scripts.spora_event_scan_selector.read_provenance_batch', return_value=rows):
            self.assertEqual(select_event(self.model, self.state, 1, 'effective_first', as_of=5)['event_time'], 0)
            self.assertEqual(select_event(self.model, self.state, 1, 'effective_latest', as_of=3)['event_time'], 1)
            self.assertEqual(select_event(self.model, self.state, 1, 'effective_previous', as_of=5)['event_time'], 1)
            self.assertEqual(select_event(self.model, self.state, 1, 'effective_latest', as_of=1)['effective_time'], 0)

    def test_effective_selection_abstains_when_time_is_missing(self):
        with patch('scripts.spora_event_scan_selector.read_provenance_batch',
                   return_value=[{'action': 'new', 'valid_at': None}]):
            self.assertEqual(select_event(self.model, self.state, 1, 'effective_latest', as_of=5)['reason'],
                             'missing_effective_time')

    def test_deferred_factual_observation_blocks_effective_selection(self):
        rows = [{'action': 'new', 'source': 'direct', 'valid_at': 0},
                {'action': 'correct', 'source': 'direct', 'valid_at': 4},
                {'action': 'defer', 'source': 'direct', 'valid_at': 2}]
        with patch('scripts.spora_event_scan_selector.read_provenance_batch', return_value=rows):
            for order in ('effective_first', 'effective_latest', 'effective_previous'):
                result = select_event(self.model, self.state, 1, order, as_of=5)
                self.assertEqual(result['reason'], 'unresolved_factual_observation')
                self.assertEqual(result['event_time'], 2)
            self.assertEqual(select_event(self.model, self.state, 1, 'effective_latest', as_of=1)['event_time'], 0)
            self.assertEqual(select_event(self.model, self.state, 1, 'previous')['event_time'], 0)

    def test_deferred_nonfactual_observation_does_not_block(self):
        rows = [{'action': 'new', 'source': 'direct', 'valid_at': 0},
                {'action': 'correct', 'source': 'direct', 'valid_at': 4},
                {'action': 'defer', 'source': 'inference', 'valid_at': 2}]
        with patch('scripts.spora_event_scan_selector.read_provenance_batch', return_value=rows):
            self.assertEqual(select_event(self.model, self.state, 1, 'effective_previous', as_of=5)['event_time'], 0)


if __name__ == '__main__':
    unittest.main()
