import unittest
from unittest.mock import patch

import torch

from spora.research_dialogue_revisions import DialogueRevisionWriter, Revision, revise
from spora.research_dialogue_working_state import characters


class DialogueRevisionTests(unittest.TestCase):
    def test_replacement_does_not_erase_other_fields(self):
        original = {'day': '周三', 'place': '北门', 'gear': '雨衣'}
        self.assertEqual(revise(original, {'place': '服务台', 'gear': None}), {'day': '周三', 'place': '服务台'})
        self.assertEqual(original, {'day': '周三', 'place': '北门', 'gear': '雨衣'})

    def test_learned_updates_use_only_current_user_spans(self):
        text = '改去西侧长廊'
        fields = ('place', 'day', 'gear')
        model = DialogueRevisionWriter(4, fields, width=8)
        ops = torch.zeros((1, 3, 3))
        ops[0, 0, Revision.SET] = 10
        ops[0, 2, Revision.REMOVE] = 10
        spans = torch.zeros((1, 3, 2, len(text) + 1))
        spans[0, 0, 0, 3] = 10
        spans[0, 0, 1, len(text)] = 10
        with patch.object(model, 'forward', return_value=(ops, spans)):
            updates = model.updates([text], characters([text]), torch.zeros((1, len(text), 4)))
        self.assertEqual(updates, [{'place': '西侧长廊', 'gear': None}])

    def test_loss_is_finite_without_any_span_writes(self):
        text = '都照旧'
        model = DialogueRevisionWriter(4, ('place', 'day'), width=8)
        ops = torch.zeros((1, 2), dtype=torch.long)
        spans = torch.zeros((1, 2, 2), dtype=torch.long)
        hidden = torch.zeros((1, len(text), 4))
        loss = model.loss(characters([text]), ops, spans, hidden)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
