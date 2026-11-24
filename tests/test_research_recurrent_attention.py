import unittest

import torch
from torch.nn import functional as F

from spora.research_recurrent_attention import RecurrentAttention


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

    def test_training_reaches_summary_and_gate(self):
        output, _ = self.model(self.q, self.k, self.v)
        output.square().mean().backward()
        for parameter in self.model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.norm().item(), 0.)
