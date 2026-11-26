import unittest

import torch
from torch.nn import functional as F

from spora.research_recurrent_attention import (
    DeltaRecurrentAttention, GatedDeltaRecurrentAttention, RecurrentAttention, SelectiveGatedDeltaAttention,
    SplitGatedDeltaRecurrentAttention,
)


class RecurrentAttentionTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(91)
        self.q, self.k, self.v = (torch.randn(1, 2, 25, 4, generator=generator) for _ in range(3))
        self.model = RecurrentAttention(2, 4, block=4, features=5)

    def test_prefill_and_irregular_stream_are_equal_with_bounded_state(self):
        expected, state = self.model(self.q, self.k, self.v)
        inputs = zip(*(tensor.split((1, 9, 4, 11), dim=-2) for tensor in (self.q, self.k, self.v)), strict=True)
        found, stream_state = [], None
        for chunk in inputs:
            output, stream_state = self.model(*chunk, state=stream_state)
            found.append(output)
            self.assertLessEqual(stream_state.key.shape[-2], 8)
            self.assertEqual(stream_state.matrix.shape, (1, 2, 5, 4))
        torch.testing.assert_close(expected, torch.cat(found, dim=-2))
        torch.testing.assert_close(state.matrix, stream_state.matrix)
        torch.testing.assert_close(state.normalizer, stream_state.normalizer)

    def test_prefix_is_exact_and_future_inputs_do_not_change_it(self):
        output, _ = self.model(self.q, self.k, self.v)
        prefix = tuple(tensor[:, :, :8] for tensor in (self.q, self.k, self.v))
        torch.testing.assert_close(output[:, :, :8], F.scaled_dot_product_attention(*prefix, is_causal=True))
        edited = self.v.clone()
        edited[:, :, 17:] += 30
        perturbed, _ = self.model(self.q, self.k, edited)
        torch.testing.assert_close(output[:, :, :17], perturbed[:, :, :17])
        self.assertFalse(torch.equal(output[:, :, 17:], perturbed[:, :, 17:]))

    def test_gqa_keeps_native_key_width_and_storage_precision(self):
        key, value = (tensor[:, :1].half() for tensor in (self.k, self.v))
        expected, _ = self.model(self.q, key.float().repeat_interleave(2, dim=1),
                                 value.float().repeat_interleave(2, dim=1))
        actual, state = self.model(self.q, key, value)
        torch.testing.assert_close(expected, actual)
        self.assertEqual(state.key.shape[1], 1)
        self.assertIs(state.key.dtype, torch.float16)
        self.assertIs(state.matrix.dtype, torch.float32)

    def test_batched_prefill_matches_sequential_and_keeps_only_recent_storage(self):
        key, value = self.k[:, :1].half(), self.v[:, :1].half()
        for count in (16, 20, 23, 25):
            with self.subTest(count=count):
                inputs = tuple(tensor[:, :, :count] for tensor in (self.q, key, value))
                for include_far in (False, True):
                    expected, state = self.model(*inputs, include_far=include_far)
                    output, fast = self.model(*inputs, include_far=include_far, batched_prefill=True)
                    torch.testing.assert_close(expected, output)
                    for name in ('key', 'value', 'matrix', 'normalizer'):
                        saved = getattr(fast, name)
                        torch.testing.assert_close(getattr(state, name), saved)
                        self.assertEqual(saved.untyped_storage().nbytes(), saved.numel() * saved.element_size())
                if count < 25:
                    continuation = tuple(tensor[:, :, count:] for tensor in (self.q, key, value))
                    slow_tail, _ = self.model(*continuation, state=state)
                    fast_tail, _ = self.model(*continuation, state=fast)
                    torch.testing.assert_close(slow_tail, fast_tail)

    def test_training_reaches_summary_and_gate(self):
        output, _ = self.model(self.q, self.k, self.v)
        output.square().mean().backward()
        for parameter in self.model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.norm().item(), 0.)

    def test_delta_revises_matching_key_and_keeps_orthogonal_binding(self):
        model = DeltaRecurrentAttention(1, 1, block=2, features=2)
        a, b = (row.repeat(1, 1, 2, 1) for row in torch.eye(2).view(2, 1, 1, 1, 2))
        matrix, norm = torch.zeros(1, 1, 2, 1), torch.zeros(1, 1, 2)
        for key, value in ((a, 3.), (b, 7.), (a, 5.)):
            matrix, norm = model.fold(key, torch.full((1, 1, 2, 1), value), matrix, norm)
        torch.testing.assert_close(matrix.flatten(), torch.tensor([5., 7.]))

    def test_delta_batch_matches_incremental_state_and_gradients_are_finite(self):
        model = DeltaRecurrentAttention(2, 4, block=4, features=5)
        inputs = (self.q, self.k[:, :1].half(), self.v[:, :1].half())
        expected, state = model(*inputs)
        output, fast = model(*inputs, batched_prefill=True)
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(fast.matrix, state.matrix)
        output.square().mean().backward()
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.norm()), 0.)

    def test_gated_delta_uses_same_state_and_trains_both_gates(self):
        model = GatedDeltaRecurrentAttention(2, 4, block=4, features=5)
        inputs = (self.q, self.k[:, :1], self.v[:, :1])
        expected, state = model(*inputs)
        output, batched = model(*inputs, batched_prefill=True)
        torch.testing.assert_close(expected, output)
        for original, saved in zip(state.tensors(), batched.tensors(), strict=True):
            torch.testing.assert_close(original, saved)
        output.square().mean().backward()
        for column in model.history_gates.grad.unbind(dim=-1):
            self.assertTrue(torch.isfinite(column).all())
            self.assertGreater(float(column.norm()), 0.)

    def test_selective_history_keeps_residual_writes_and_native_kv_storage(self):
        inputs = (self.q, self.k[:, :1].half(), torch.zeros_like(self.v[:, :1]).half())
        inputs[2][:, :, 1, :] = 100.
        model = SelectiveGatedDeltaAttention(2, 4, block=4, features=5, retention_slots=3, policy='residual')
        expected, final = model(*inputs, batched_prefill=True)
        state, found = None, []
        chunks = zip(*(tensor.split((1, 9, 4, 11), dim=-2) for tensor in inputs), strict=True)
        for chunk in chunks:
            output, state = model(*chunk, state=state)
            found.append(output)
            self.assertLessEqual(state.key.shape[-2], 8)
        torch.testing.assert_close(expected, torch.cat(found, dim=-2))
        for original, streamed in zip(final.tensors(), state.tensors(), strict=True):
            torch.testing.assert_close(original, streamed)
        retained = final.retained
        self.assertEqual(retained.key.shape, (1, 1, 3, 4))
        self.assertEqual(retained.key.dtype, torch.float16)
        self.assertIn(1, retained.positions[0, 0].tolist())
        self.assertLess(int(retained.positions.max()), final.length - final.key.shape[-2])
        for tensor in retained.tensors():
            self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size())

    def test_recent_slots_reproduce_full_causal_attention_while_every_entry_fits(self):
        model = SelectiveGatedDeltaAttention(2, 4, block=4, features=5, retention_slots=16, policy='recency')
        inputs = tuple(tensor[:, :, :23] for tensor in (self.q, self.k, self.v))
        actual, state = model(*inputs, include_far=False)
        torch.testing.assert_close(actual, F.scaled_dot_product_attention(*inputs, is_causal=True))
        self.assertEqual(state.retained.positions[0, 0].tolist(), list(range(15, -1, -1)))

    def test_split_delta_starts_tied_and_erases_only_along_matching_keys(self):
        tied = GatedDeltaRecurrentAttention(1, 1, block=2, features=2)
        split = SplitGatedDeltaRecurrentAttention(1, 1, block=2, features=2)
        missing, unexpected = split.load_state_dict(tied.state_dict(), strict=False)
        self.assertEqual(set(missing), {'erase_offset', 'erase_bias'})
        self.assertEqual(unexpected, [])
        keys = torch.tensor([[[[1., 0.], [1., 0.]]]])
        matrix = torch.tensor([[[[3.], [7.]]]])
        norm = torch.zeros(1, 1, 2)
        values = torch.full((1, 1, 2, 1), 5.)
        original, _ = tied.fold(keys, values, matrix, norm)
        initialized, _ = split.fold(keys, values, matrix, norm)
        torch.testing.assert_close(initialized, original)
        initialized.sum().backward()
        self.assertGreater(split.erase_offset.grad.norm().item(), 0.)
        with torch.no_grad():
            split.erase_bias.fill_(-10.)
            less_erased, _ = split.fold(keys, values, matrix, norm)
            empty_written, _ = split.fold(keys, values, torch.zeros_like(matrix), norm)
            tied_written, _ = tied.fold(keys, values, torch.zeros_like(matrix), norm)
        self.assertGreater(float(less_erased[0, 0, 0, 0]), float(original.detach()[0, 0, 0, 0]))
        torch.testing.assert_close(less_erased[:, :, 1], original[:, :, 1])
        torch.testing.assert_close(empty_written, tied_written)

    def test_unrotated_history_keeps_local_attention_and_matches_stream(self):
        key, value = self.k[:, :1].half(), self.v[:, :1].half()
        history = (-self.q, -key)
        for core in (self.model, DeltaRecurrentAttention(2, 4, block=4, features=5),
                     SplitGatedDeltaRecurrentAttention(2, 4, block=4, features=5)):
            full, final = core(self.q, key, value, history=history, batched_prefill=True)
            native, _ = core(self.q, key, value, include_far=False, batched_prefill=True)
            local, _ = core(self.q, key, value, history=history, include_far=False, batched_prefill=True)
            torch.testing.assert_close(local, native)
            torch.testing.assert_close(full[:, :, :8], native[:, :, :8])
            self.assertFalse(torch.equal(full[:, :, 8:], native[:, :, 8:]))
            cached, state = core(self.q[:, :, :17], key[:, :, :17], value[:, :, :17],
                                 history=tuple(tensor[:, :, :17] for tensor in history), batched_prefill=True)
            tail, streamed = core(self.q[:, :, 17:], key[:, :, 17:], value[:, :, 17:], state=state,
                                  history=tuple(tensor[:, :, 17:] for tensor in history))
            torch.testing.assert_close(full, torch.cat((cached, tail), dim=2))
            for expected, actual in zip(final.tensors(), streamed.tensors(), strict=True):
                torch.testing.assert_close(expected, actual)
            self.assertLessEqual(final.history_key.shape[-2], 2 * core.block)
            with self.assertRaisesRegex(ValueError, 'changed during the stream'):
                core(self.q[:, :, 17:], key[:, :, 17:], value[:, :, 17:], state=state)
