import unittest
import torch

from spora.memory_read import MemoryReadPlan
from spora.language_bridge import LanguageBridge
from spora.temporal import TemporalSPORA, TemporalConfig
from spora.growing import new_state, observe, grow


def reference(matrices, queries, keys):
    values = torch.stack([torch.einsum('bmd,bnd->bnm', m.expand(queries.shape[0], -1, -1), queries)
                          for m in matrices], 2)
    scores = ((values * queries.unsqueeze(2)).sum(-1) +
              torch.einsum('bnm,pm->bnp', queries, keys)) / queries.shape[-1]**.5
    return (values * scores.softmax(-1).unsqueeze(-1)).sum(2)


class MemoryReadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2110000)
        torch.set_num_threads(2)

    @torch.no_grad()
    def test_identical_pages_keep_distinct_position_weights(self):
        a, b = torch.randn(1, 12, 12), torch.randn(1, 12, 12)
        matrices = [a, b, b.clone(), b.clone()]
        q, keys = torch.randn(2, 9, 12), torch.randn(4, 12)
        plan = MemoryReadPlan(matrices)
        actual, stats = plan.read(q, keys)
        torch.testing.assert_close(actual, reference(matrices, q, keys), atol=1e-5, rtol=1e-5)
        self.assertEqual(stats['unique_matrices'], 2)
        self.assertEqual(stats['matrix_vectors'], 36)

    @torch.no_grad()
    def test_selective_attention_reads_one_page_with_bounded_error(self):
        matrices = [torch.randn(1, 16, 16) * .05 for _ in range(9)]
        q = torch.zeros(2, 6, 16); q[..., 0] = 1
        keys = torch.zeros(9, 16); keys[4, 0] = 100
        actual, stats = MemoryReadPlan(matrices).read(q, keys, 'bounded')
        error = (actual-reference(matrices, q, keys)).norm(dim=-1).max()
        self.assertLessEqual(float(error), stats['max_read_error_bound'] + 1e-6)
        self.assertEqual(stats['pruned_queries'], 12)
        self.assertEqual(stats['matrix_vectors'], 12)

    @torch.no_grad()
    def test_ambiguous_attention_falls_back_without_forced_topk(self):
        matrices = [torch.randn(1, 8, 8) for _ in range(8)]
        q = torch.nn.functional.normalize(torch.randn(2, 7, 8), dim=-1)
        keys = torch.zeros(8, 8)
        actual, stats = MemoryReadPlan(matrices).read(q, keys, 'bounded')
        torch.testing.assert_close(actual, reference(matrices, q, keys), atol=1e-6, rtol=1e-5)
        self.assertEqual(stats['fallback_queries'], 14)

    @torch.no_grad()
    def test_large_unselected_content_cannot_be_hidden_by_position_score(self):
        matrices = [torch.randn(1, 8, 8) * .01 for _ in range(4)]
        matrices[3][0, 0, 0] = 180
        q = torch.zeros(1, 3, 8); q[..., 0] = 1
        keys = torch.zeros(4, 8); keys[0, 0] = 60
        actual, stats = MemoryReadPlan(matrices).read(q, keys, 'bounded')
        expected = reference(matrices, q, keys)
        self.assertGreater(float(expected[..., 0].min()), 179)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)

    @torch.no_grad()
    def test_off_diagonal_information_prevents_unsafe_pruning(self):
        matrices = [torch.zeros(1, 8, 8) for _ in range(4)]
        matrices[3][0, 0, 1] = 300
        q = torch.zeros(1, 3, 8); q[..., :2] = 2**-.5
        keys = torch.zeros(4, 8); keys[0, 0] = 90
        actual, stats = MemoryReadPlan(matrices).read(q, keys, 'bounded')
        self.assertEqual(stats['fallback_queries'], 3)
        torch.testing.assert_close(actual, reference(matrices, q, keys), atol=1e-4, rtol=1e-5)

    def test_derived_state_cannot_silently_cut_training_gradients(self):
        plan = MemoryReadPlan([torch.eye(8)[None]])
        with self.assertRaises(RuntimeError):
            plan.read(torch.randn(1, 2, 8, requires_grad=True), torch.zeros(1, 8))

    @torch.no_grad()
    def test_direct_plan_rejects_stale_matrices(self):
        matrix = torch.eye(8)[None]
        plan = MemoryReadPlan([matrix])
        matrix.add_(1)
        with self.assertRaises(ValueError):
            plan.read(torch.randn(1, 2, 8), torch.zeros(1, 8))

    def test_inference_mode_without_tensor_versions_uses_live_reads(self):
        with torch.inference_mode():
            model = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2,
                                  memory_width=16, expert_rank=4, memory_layers=(0, 1),
                                  history_writer='rls'))
            state = grow(model, new_state(model), 3)
            bridge = LanguageBridge(model)
            tokens = torch.randint(0, 256, (1, 8))
            before = bridge(tokens, state)
            self.assertTrue(all(s['mode'] == 'legacy-unversioned' for s in bridge.last_read_stats))
            state.current[0].add_(1)
            actual = bridge(tokens, state)
            bridge.read_mode = 'legacy'
            torch.testing.assert_close(actual, bridge(tokens, state), atol=0, rtol=0)
            self.assertFalse(torch.equal(before, actual))

    @torch.no_grad()
    def test_updates_and_growth_invalidate_plan(self):
        model = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2,
                              memory_width=16, expert_rank=4, memory_layers=(0, 1),
                              history_writer='rls'))
        state = grow(model, observe(model, new_state(model), 2, 5, 1), 3)
        bridge = LanguageBridge(model)
        tokens = torch.randint(0, 256, (1, 8))
        bridge(tokens, state)
        self.assertTrue(all(x['plan_rebuilt'] for x in bridge.last_read_stats))
        bridge(tokens, state)
        self.assertFalse(any(x['plan_rebuilt'] for x in bridge.last_read_stats))
        state.current[0].add_(.1)
        updated = bridge(tokens, state)
        self.assertTrue(bridge.last_read_stats[0]['plan_rebuilt'])
        bridge.read_mode = 'legacy'
        torch.testing.assert_close(updated, bridge(tokens, state), atol=1e-5, rtol=1e-5)
        bridge.read_mode = 'exact'
        state = observe(model, state, 3, 6, 1)
        bridge(tokens, state)
        self.assertTrue(all(x['plan_rebuilt'] for x in bridge.last_read_stats))
        state = grow(model, state)
        bridge(tokens, state)
        self.assertTrue(all(x['plan_rebuilt'] for x in bridge.last_read_stats))


if __name__ == '__main__':
    unittest.main()
