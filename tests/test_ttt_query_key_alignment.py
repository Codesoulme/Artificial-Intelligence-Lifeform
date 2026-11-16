import unittest

import torch
from torch import nn

from scripts.deepseek_spora_ttt_query_key_alignment import ridge_map
from spora.research_ttt_ntp import TTTNTPMemoryLinear


class QueryKeyAlignmentTests(unittest.TestCase):
    def test_ridge_map_recovers_a_well_conditioned_linear_map(self):
        torch.manual_seed(4)
        x = torch.randn(20, 4)
        truth = torch.randn(4, 3)
        y = x @ truth
        fitted = ridge_map(x, y, ridge=1e-6)
        torch.testing.assert_close(x @ fitted, y, atol=1e-3, rtol=1e-3)

    def test_rejects_mismatched_rows(self):
        with self.assertRaises(ValueError):
            ridge_map(torch.zeros(2, 3), torch.zeros(1, 3))

    def test_query_capture_hook_does_not_replace_down_projection_output(self):
        base = nn.Linear(4, 2, bias=False)
        layer = TTTNTPMemoryLinear(base, 0)
        captured = {}

        def capture(module, args, output):
            captured['query'] = args[0][0, -1].detach()

        handle = layer.base.register_forward_hook(capture)
        try:
            hidden = torch.randn(1, 3, 4)
            result = layer(hidden)
        finally:
            handle.remove()
        self.assertEqual(tuple(result.shape), (1, 3, 2))
        self.assertEqual(tuple(captured['query'].shape), (4,))


if __name__ == '__main__':
    unittest.main()
