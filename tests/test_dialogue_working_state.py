from contextvars import ContextVar
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from spora.research_dialogue_working_state import (
    FIELDS, JointTurnValueWriter, TurnValueWriter, WorkingTurnReader, WorkingTurnState, characters,
)


class TinyDecoder:
    def __init__(self):
        self.base = SimpleNamespace(device=torch.device('cpu'), model=SimpleNamespace(layers=[nn.Identity()]),
                                    lm_head=nn.Linear(8, 32, bias=False))
        self.embedding = nn.Embedding(32, 8)
        self.base.get_input_embeddings = lambda: self.embedding
        self._memory_context = ContextVar('test_durable_memory', default=None)

    def logits(self):
        features = self.embedding(torch.tensor([[1, 2, 3]]))
        for layer in self.base.model.layers:
            features = layer(features)
        return self.base.lm_head(features)


class DialogueWorkingStateTests(unittest.TestCase):
    def test_update_overwrites_only_the_written_field(self):
        base = WorkingTurnState().update({'day': (1, 2), 'place': (4, 5), 'time': (7,)})
        nxt = base.update({'time': (8, 9)})
        self.assertEqual(base.slots[FIELDS.index('time')], (7,))
        self.assertEqual(nxt.slots[FIELDS.index('time')], (8, 9))
        self.assertEqual(nxt.slots[FIELDS.index('place')], (4, 5))
        self.assertEqual(nxt.update({}), nxt)

    def test_writer_has_separate_absent_and_span_positions(self):
        writer = TurnValueWriter(8, feature_width=6)
        codes = characters(['下雨', '请更改地址'])
        output = writer(codes, torch.ones(*codes.shape, 6))
        self.assertEqual(output.shape, (2, len(FIELDS), 2, 6))
        self.assertTrue(torch.isfinite(output[..., 0]).all())
        self.assertTrue(torch.isneginf(output[0, ..., 3:]).all())

    def test_joint_writer_scores_only_complete_valid_spans_and_no_write(self):
        writer = JointTurnValueWriter(8, feature_width=6)
        codes = characters(['去北门', '请更改地址'])
        features = torch.ones(*codes.shape, 6)
        width = codes.shape[1]
        scores = writer(codes, features)
        self.assertEqual(scores.shape, (2, len(FIELDS), 1 + width ** 2))
        self.assertTrue(torch.isfinite(scores[..., 0]).all())
        self.assertTrue(torch.isfinite(scores[0, :, 1 + width + 2]).all())
        self.assertTrue(torch.isneginf(scores[0, :, 1 + 2 * width + 1]).all())
        self.assertTrue(torch.isneginf(scores[0, :, 1 + width + 3]).all())
        boundaries = torch.zeros(2, len(FIELDS), 2, dtype=torch.long)
        boundaries[0, FIELDS.index('place')] = torch.tensor([2, 3])
        loss = writer.loss(codes, boundaries, features)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(writer.pair.weight.grad).all())
        self.assertGreater(writer.pair.weight.grad.norm().item(), 0.)

    def test_joint_writer_uses_no_write_even_with_a_legal_span(self):
        writer = JointTurnValueWriter(8)
        scores = torch.full((1, len(FIELDS), 17), -10.)
        scores[..., 0] = 5.
        place = FIELDS.index('place')
        scores[0, place, 1 + 4 + 3] = 6.
        writer.forward = lambda codes, features: scores
        self.assertEqual(writer.write('到青芒站', margin=2.), {})
        self.assertEqual(writer.write('到青芒站', margin=.5), {'place': '青芒站'})

    def test_reader_starts_neutral_and_excludes_durable_reads(self):
        decoder = TinyDecoder()
        baseline = decoder.logits()
        state = WorkingTurnState().update({'time': (9, 12)})
        reader = WorkingTurnReader(8, width=4)
        with reader.active(decoder, state):
            self.assertTrue(torch.equal(decoder.logits(), baseline))
        with torch.no_grad():
            reader.output.weight.fill_(.5)
            reader.copy_scale.fill_(1)
        with reader.active(decoder, state):
            self.assertFalse(torch.equal(decoder.logits(), baseline))
            token = decoder._memory_context.set(object())
            try:
                self.assertTrue(torch.equal(decoder.logits(), baseline))
            finally:
                decoder._memory_context.reset(token)
        self.assertTrue(torch.equal(decoder.logits(), baseline))

    def test_copy_guidance_is_differentiable_and_hooks_close_on_failure(self):
        decoder = TinyDecoder()
        baseline = decoder.logits()
        reader = WorkingTurnReader(8, width=4)
        state = WorkingTurnState().update({'place': (9, 12)})
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            with reader.active(decoder, state, targets=torch.tensor([9, 12, 2])) as losses:
                decoder.logits()
                self.assertEqual(len(losses), 1)
                losses[0].backward()
                self.assertGreater(reader.query.weight.grad.norm().item(), 0.)
                raise RuntimeError('cancelled')
        self.assertTrue(torch.equal(decoder.logits(), baseline))

    def test_internal_read_at_selected_layer_with_copy_scores_disabled(self):
        decoder = TinyDecoder()
        decoder.base.model.layers.append(nn.Linear(8, 8, bias=False))
        with torch.no_grad():
            decoder.base.model.layers[1].weight.zero_()
        baseline = decoder.logits()
        reader = WorkingTurnReader(8, width=4, layer_index=0, copy_logits=False)
        state = WorkingTurnState().update({'time': (9, 12)})
        with torch.no_grad():
            reader.output.weight.fill_(.5)
            reader.copy_scale.fill_(2.)
        with reader.active(decoder, state, targets=torch.tensor([9, 12, 2])) as losses:
            self.assertTrue(torch.equal(decoder.logits(), baseline))
            self.assertEqual(len(losses), 1)
            losses[0].backward()
            self.assertTrue(torch.isfinite(reader.key.weight.grad).all())
        with torch.no_grad():
            decoder.base.model.layers[1].weight.copy_(torch.eye(8))
        baseline = decoder.logits()
        with reader.active(decoder, state):
            self.assertFalse(torch.equal(decoder.logits(), baseline))
        self.assertTrue(torch.equal(decoder.logits(), baseline))
