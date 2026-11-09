import unittest
from unittest.mock import patch

import torch

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_full_answer import answer_batch
from scripts.deepseek_learned_address import select_addresses
from scripts.deepseek_pooled_address import PooledAddress, question_mask
from scripts.deepseek_memory_intent import IntentGate, build_cases
from spora.open_base_memory import OpenBaseMemory


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

    def test_learned_selector_uses_each_site_and_propagates_gradients(self):
        adapter = OpenBaseMemory(3, 3)
        with torch.no_grad():
            for projection in adapter.query:
                projection.weight.copy_(torch.eye(3))
        features = torch.tensor([[[2., 0., 0.], [0., 3., 0.]]])
        addresses = select_addresses(adapter, features)
        self.assertTrue(torch.allclose(addresses[0], torch.tensor([[1., 0., 0.]])))
        self.assertTrue(torch.allclose(addresses[1], torch.tensor([[0., 1., 0.]])))
        (addresses[0][0, 1] + addresses[1][0, 0]).backward()
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.query.parameters()))
        with self.assertRaises(ValueError):
            select_addresses(adapter, torch.zeros(1, 3, 3))

    def test_pooled_selector_ignores_padding_and_rejects_empty_mask(self):
        selector = PooledAddress(3, 3)
        with torch.no_grad():
            for query in selector.query:
                query.weight.copy_(torch.eye(3))
        features = torch.tensor([[[[2., 0., 0.], [0., 100., 0.]],
                                  [[0., 3., 0.], [100., 0., 0.]]]])
        addresses = selector(features, torch.tensor([[True, False]]))
        self.assertTrue(torch.allclose(addresses[0], torch.tensor([[1., 0., 0.]])))
        self.assertTrue(torch.allclose(addresses[1], torch.tensor([[0., 1., 0.]])))
        with self.assertRaises(ValueError):
            selector(features, torch.tensor([[False, False]]))

    def test_question_mask_excludes_chat_scaffolding_and_padding(self):
        offsets = torch.tensor([[[0, 2], [2, 3], [3, 5], [5, 8], [0, 0]]])
        mask = question_mask(offsets, torch.tensor([[1, 1, 1, 1, 0]]), ['U:foo A:'], ['foo'])
        self.assertEqual(mask.tolist(), [[False, True, True, False, False]])
        with self.assertRaises(ValueError):
            question_mask(offsets, torch.ones(1, 5), ['foo foo'], ['foo'])

    def test_intent_dataset_splits_and_question_mask(self):
        for augmented, malformed in ((False, False), (True, False), (True, True)):
            training = build_cases(augmented=augmented, malformed=malformed)
            held = build_cases(True, augmented=augmented, malformed=malformed)
            self.assertFalse({row['question'] for row in training} & {row['question'] for row in held})
            self.assertEqual({row['label'] for row in training}, {0, 1})
        gate = IntentGate(3)
        features = torch.randn(2, 2, 3, 3)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        original = gate(features, mask)
        changed = features.clone()
        changed[~mask[:, None, :, None].expand_as(changed)] = 1000
        self.assertTrue(torch.allclose(original, gate(changed, mask)))
