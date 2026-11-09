import unittest
from unittest.mock import patch

import torch

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_full_answer import answer_batch


class FakeBatch:
    def __init__(self, text):
        self.text = text
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    def __init__(self, template='<BOS><user>question<assistant><think>'):
        self.template = template

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert messages == [{'role': 'user', 'content': 'question'}]
        assert tokenize is False and add_generation_prompt is True
        return self.template

    def __call__(self, text, return_tensors, add_special_tokens):
        assert return_tensors == 'pt' and add_special_tokens is False
        return FakeBatch(text)


class DeepseekReadPrefixTests(unittest.TestCase):
    def test_prefix_closes_thinking_without_inserting_an_answer(self):
        batch = closed_think_inputs(FakeTokenizer(), 'question', 'cpu')
        self.assertEqual(batch.text, '<BOS><user>question<assistant><think></think>\n\n')
        self.assertEqual(batch.device, 'cpu')

    def test_unexpected_chat_template_is_rejected(self):
        for template in ('no reasoning prefix', '<think>already closed</think>'):
            with self.assertRaises(ValueError):
                closed_think_inputs(FakeTokenizer(template), 'question', 'cpu')

    def test_answer_supervision_includes_eos_and_masks_padding(self):
        class NumericTokenizer:
            eos_token_id = 9
            pad_token_id = 0

            def encode(self, text, add_special_tokens):
                return [int(char) + 1 for char in text]

        with patch('scripts.deepseek_full_answer.closed_think_inputs',
                   return_value={'input_ids': torch.tensor([[7, 8]])}):
            inputs, labels = answer_batch(NumericTokenizer(), 'question', [3, 15], 'cpu')
        self.assertEqual(inputs['input_ids'].tolist(), [[7, 8, 4, 9, 0], [7, 8, 2, 6, 9]])
        self.assertEqual(labels.tolist(), [[-100, -100, 4, 9, -100], [-100, -100, 2, 6, 9]])
        self.assertEqual(inputs['attention_mask'].tolist(), [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])

    def test_thinking_format_supervises_closing_marker_and_answer(self):
        class ReasoningTokenizer:
            eos_token_id = 9
            pad_token_id = 0

            def encode(self, text, add_special_tokens):
                self.text = text
                return [3, 4, 5]

        tokenizer = ReasoningTokenizer()
        with patch('scripts.deepseek_full_answer.thinking_inputs',
                   return_value={'input_ids': torch.tensor([[7, 8]])}):
            inputs, labels = answer_batch(tokenizer, 'question', [2], 'cpu', 'check\n')
        self.assertEqual(tokenizer.text, 'check\n</think>\n\n2')
        self.assertEqual(inputs['input_ids'].tolist(), [[7, 8, 3, 4, 5, 9]])
        self.assertEqual(labels.tolist(), [[-100, -100, 3, 4, 5, 9]])
