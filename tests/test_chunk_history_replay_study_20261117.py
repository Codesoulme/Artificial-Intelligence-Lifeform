import unittest

import torch

from scripts.deepseek_chunk_history_replay_study_20261117 import solve_replay


class ReplaySolveTests(unittest.TestCase):
    def test_matches_small_closed_form_and_is_float32(self):
        q = torch.tensor([[1., 0.], [0., 2.], [1., 1.]])
        y = torch.tensor([[1., 3.], [2., 4.], [3., 1.]])
        ridge = 0.1
        got = solve_replay(q, y, ridge)
        expected = torch.linalg.solve(q.double().T @ q.double() + ridge * torch.eye(2, dtype=torch.float64),
                                      q.double().T @ y.double()).T.float()
        torch.testing.assert_close(got, expected)
        self.assertEqual(got.dtype, torch.float32)

    def test_rejects_invalid_and_nonfinite_inputs(self):
        q = torch.eye(2)
        with self.assertRaises(ValueError):
            solve_replay(q, torch.ones(3, 2), 0.01)
        with self.assertRaises(ValueError):
            solve_replay(q, q, 0.)
        q[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            solve_replay(q, q, 0.01)
