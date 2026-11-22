from contextvars import ContextVar
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from spora.research_dialogue_working_state import FIELDS, TurnValueWriter, WorkingTurnReader, WorkingTurnState, characters


class TinyDecoder:
    def __init__(self):
        self.base = SimpleNamespace(device=torch.device('cpu'), model=SimpleNamespace(layers=[nn.Identity()]),
                                    lm_head=nn.Linear(8, 32, bias=False))
        self.embedding = nn.Embedding(32, 8)
        self.base.get_input_embeddings = lambda: self.embedding
        self._memory_context = ContextVar('test_durable_memory', default=None)

    def logits(self):
        features = self.embedding(torch.tensor([[1, 2, 3]]))
        return self.base.lm_head(self.base.model.layers[0](features))


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
