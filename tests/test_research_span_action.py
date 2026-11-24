import unittest

import torch

from spora.research_dialogue_working_state import FIELDS, WorkingTurnState
from spora.research_recurrent_attention import is_append_only_causal_mask
from spora.research_span_action import SpanActionHead, available_fields, compile_actions, emit_action


class SpanActionTests(unittest.TestCase):
    def test_whole_field_action_expands_tokens_and_consumes_one_position(self):
        place, person = FIELDS.index('place'), FIELDS.index('person')
        state = WorkingTurnState().update({'place': (2, 4, 6), 'person': (9,)})
        vocab = 32
        surface, targets = compile_actions([(None, (1,)), (place, (2, 4, 6)), (None, (3,)),
                                            (person, (9,))], state, vocab, 10)
        self.assertEqual(surface, [1, 2, 4, 6, 3, 9])
        self.assertEqual([item.offset for item in targets], [0, 1, 4, 5, 6])
        self.assertEqual([item.action for item in targets], [1, vocab + place, 3, vocab + person, 10])
        allowed = available_fields(state, targets, vocab, 'cpu')
        self.assertEqual(allowed[:, place].tolist(), [True, True, False, False, False])
        self.assertEqual(allowed[:, person].tolist(), [True, True, True, True, False])
        remaining = {place, person}
        with self.assertRaises(ValueError):
            emit_action(vocab + place, state, remaining, 2, vocab)
        self.assertEqual(emit_action(vocab + place, state, remaining, 3, vocab), (2, 4, 6))
        self.assertEqual(remaining, {person})
        with self.assertRaises(ValueError):
            emit_action(vocab + place, state, remaining, 3, vocab)

    def test_native_words_remain_available_without_copy_actions(self):
        torch.manual_seed(21)
        reader = SpanActionHead(8, width=4)
        embeddings = torch.nn.Embedding(16, 8)
        language = torch.nn.Linear(8, 16, bias=False)
        state = WorkingTurnState().update({'place': (2, 4, 6)})
        field = FIELDS.index('place')
        keys = reader.field_keys(state, embeddings)
        hidden = torch.randn(2, 8)
        available = torch.zeros(2, len(FIELDS), dtype=torch.bool)
        available[0, field] = True
        logits = reader(hidden, keys, available, language)
        torch.testing.assert_close(logits.exp().sum(dim=-1), torch.ones(2))
        torch.testing.assert_close(logits[1, :16], language(hidden[1]).log_softmax(dim=-1))
        self.assertTrue(torch.isneginf(logits[1, 16:]).all())
        self.assertTrue(torch.isfinite(logits[0, 16 + field]))
        (-logits[0, 16 + field] - logits[1, 3]).backward()
        self.assertTrue(torch.isfinite(reader.mode.weight.grad).all())

    def test_append_mask_recognizes_causality_and_rejects_padding_and_future_tokens(self):
        pos = torch.tensor([3, 4])
        mask = torch.tensor([[[[True, True, True, True, False], [True, True, True, True, True]]]])
        self.assertTrue(is_append_only_causal_mask(mask, pos, 2))
        additive = torch.zeros_like(mask, dtype=torch.float16).masked_fill(~mask, -torch.inf)
        self.assertTrue(is_append_only_causal_mask(additive, pos, 2))
        self.assertFalse(is_append_only_causal_mask(mask, torch.tensor([2, 4]), 2))
        self.assertFalse(is_append_only_causal_mask(torch.ones_like(mask), pos, 2))
        mask[..., 0] = False
        self.assertFalse(is_append_only_causal_mask(mask, pos, 2))


if __name__ == '__main__':
    unittest.main()
