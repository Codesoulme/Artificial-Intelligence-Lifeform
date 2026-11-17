import unittest
from types import SimpleNamespace
import torch
from torch import nn
from scripts import deepseek_spora_bridge_reload_audit as audit
from spora.research_continuous_memory_attention import ContinuousMemoryAttention


class Context:
    def get(self):
        return None


class ContinuousAttentionReloadTests(unittest.TestCase):
    def test_restores_representation_gain_and_weights(self):
        context = Context()
        codec = SimpleNamespace(memory_width=12, slot_width=4, slot_count=3)
        # Audit uses four conditioned slots; give the toy codec four slots.
        codec.slot_count = 4
        codec.slot_width = 3
        source = ContinuousMemoryAttention(nn.Linear(6, 6, bias=False), context, 12, codec,
                                           None, slots=4, width=4, output_gain=10.)
        saved = {'width': 4, 'state': source.memory_state()}
        attention = SimpleNamespace(o_proj=nn.Linear(6, 6, bias=False))
        decoder = SimpleNamespace(_memory_context=context)
        model = SimpleNamespace(config=SimpleNamespace(memory_width=12))
        restored = audit.restore_cross_attention(saved, attention, decoder, model, codec, None,
            {'cross_attention_representation': 'continuous', 'cross_attention_output_gain': 10.})
        self.assertEqual(restored.output_gain, 10.)
        torch.testing.assert_close(restored.key.weight, source.key.weight)
        torch.testing.assert_close(restored.out.weight, source.out.weight)
        with self.assertRaises(ValueError):
            audit.restore_cross_attention(saved, attention, decoder, model, codec, None,
                                          {'cross_attention_representation': 'unknown'})

    def test_value_specific_question_fails_closed(self):
        previous = audit.QUESTION_BY_VALUE
        try:
            audit.QUESTION_BY_VALUE = ('请问{name}饮品？', '请问{name}颜色？')
            self.assertEqual(audit.question_template(1), '请问{name}颜色？')
            with self.assertRaises(ValueError):
                audit.question_template(2)
        finally:
            audit.QUESTION_BY_VALUE = previous
