import unittest

import torch

from spora.research_dialogue_working_state import FIELDS, characters
from spora.research_revision_writer import SeparateTurnValueWriter


class SeparateTurnValueWriterTests(unittest.TestCase):
    def test_write_gate_and_full_span_have_independent_scores(self):
        writer = SeparateTurnValueWriter(8)
        field = FIELDS.index('place')
        op = torch.full((1, len(FIELDS)), -2.)
        spans = torch.full((1, len(FIELDS), 16), -5.)
        spans[0, field, 6] = 4.
        spans[0, field, 5] = 2.
        writer.scores = lambda codes, features: (op, spans)
        self.assertEqual(writer.write('改成北门'), {})
        op[0, field] = 2.
        self.assertEqual(writer.write('改成北门'), {'place': '成北'})
        self.assertEqual(writer.write('改成北门', margin=3.), {})
        spans[0, field, 11] = 5.
        self.assertEqual(writer.write('改成北门'), {'place': '北门'})

    def test_unchanged_turn_trains_the_gate_and_positive_turn_trains_the_span(self):
        writer = SeparateTurnValueWriter(8)
        codes = characters(['改成北门', '什么地方'])
        boundaries = torch.zeros(2, len(FIELDS), 2, dtype=torch.long)
        writer.loss(codes, boundaries).backward()
        self.assertIsNotNone(writer.activity.weight.grad)
        self.assertIsNone(writer.pair.weight.grad)
        writer.zero_grad(set_to_none=True)
        boundaries[0, FIELDS.index('place')] = torch.tensor([3, 4])
        objective = writer.loss(codes, boundaries)
        objective.backward()
        self.assertTrue(torch.isfinite(objective))
        self.assertTrue(torch.isfinite(writer.pair.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
