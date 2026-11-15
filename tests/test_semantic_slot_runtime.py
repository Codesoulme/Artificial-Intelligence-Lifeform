import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from spora.semantic_slot_runtime import (NaturalSemanticSlotMemory, parse_fact,
                                          parse_question, parse_question_route)
from spora.semantic_slot_state import SemanticSlotState


class SemanticSlotRuntimeTests(unittest.TestCase):
    def test_parsers_accept_unseen_value_text_without_a_lexicon(self):
        self.assertEqual(parse_fact('小林喜欢喝一种新饮品。').value, '一种新饮品')
        self.assertEqual(parse_question('小林喜欢喝什么饮料？'), '小林')
        self.assertEqual(parse_fact('小林喜欢的颜色是蓝色。').relation, 'color')
        self.assertEqual(parse_question_route('小林喜欢什么颜色？'), ('小林', 'color'))
        with self.assertRaises(ValueError):
            parse_fact('小林喜欢喝什么饮料？')

    def test_write_replace_query_and_roundtrip(self):
        state = SemanticSlotState(3, 2)
        memory = NaturalSemanticSlotMemory(state, base_sha256='base', bridge_sha256='bridge')
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [0] if text != '乙' else [1])
        embedding = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        self.assertEqual(memory.write('甲喜欢喝乙。', tokenizer, embedding)['action'], 'new')
        self.assertEqual(memory.query_slot('甲喜欢喝什么饮料？')['slot'], 0)
        self.assertEqual(memory.write('甲喜欢的颜色是乙。', tokenizer, embedding)['action'], 'new')
        self.assertEqual(memory.query_slot('甲喜欢什么颜色？')['slot'], 1)
        self.assertEqual(memory.write('甲喜欢喝乙。', tokenizer, embedding)['action'], 'replace')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'memory.pt'
            memory.save(path)
            restored = NaturalSemanticSlotMemory.load(
                path, expected_base_sha256='base', expected_bridge_sha256='bridge')
            torch.testing.assert_close(restored.query_slot('甲喜欢喝什么饮料？')['value'], embedding[1])
            self.assertIsNone(restored.query_slot('丙喜欢喝什么饮料？')['slot'])


if __name__ == '__main__':
    unittest.main()
