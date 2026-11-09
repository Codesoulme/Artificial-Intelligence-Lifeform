import unittest
from unittest.mock import patch

from scripts.open_base_trajectory_comparison import build_trajectories, numeric_correct


class TrajectoryComparisonTests(unittest.TestCase):
    def test_trajectories_keep_corrections_and_interference(self):
        def observe(_model, state, key, value, _source, _quality):
            state[key] = value
            return state, {'action': 'write', 'current_write': True}

        original = {99: 3}
        with patch('scripts.open_base_trajectory_comparison.observe_sourced', side_effect=observe):
            trajectories = build_trajectories(None, original)
        self.assertEqual(original, {99: 3})
        self.assertEqual(len(trajectories), 4)
        for item in trajectories:
            self.assertEqual(len(item['events']), 14)
            self.assertEqual(sum(e['role'] == 'correction' for e in item['events']), 2)
            self.assertEqual(sum(e['role'] == 'interference' for e in item['events']), 8)
            for key in item['keys']:
                writes = [e['value'] for e in item['events'] if e['key'] == key]
                self.assertEqual(item['targets'][key], writes[-1])
                self.assertEqual(item['state'][key], writes[-1])
                if len(writes) == 2:
                    self.assertNotEqual(writes[0], writes[1])

    def test_exact_scoring_does_not_accept_number_as_substring(self):
        self.assertTrue(numeric_correct('7。', 7))
        self.assertFalse(numeric_correct('17', 7))
        self.assertFalse(numeric_correct('值是7', 7))
