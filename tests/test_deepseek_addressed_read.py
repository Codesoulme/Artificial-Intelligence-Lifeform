import unittest
from unittest.mock import patch

import torch

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_full_answer import answer_batch
from scripts.deepseek_learned_address import select_addresses
from scripts.deepseek_pooled_address import PooledAddress, extract, question_mask
from scripts.deepseek_memory_intent import IntentGate, build_cases
from scripts.deepseek_guard_adversarial import SECOND_CASES
from scripts.deepseek_semantic_gate import hide_numbers, unique_semantic_cases
from scripts.deepseek_composite_probe import initial_route
from scripts.deepseek_multitype_gate import MultiIntentGate, CLASSES, unique_cases, checkpoint_payload
from scripts.deepseek_route_hardening import evaluate as evaluate_hardened_gate
from spora.open_base_memory import OpenBaseMemory
from spora.finite_question_guard import valid_finite_question_form


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
        with self.assertRaises(ValueError):
            extract(None, None, [], hidden_indices=(6, 6))

    def test_intent_dataset_splits_and_question_mask(self):
        for augmented, malformed, contrastive in ((False, False, False), (True, False, False),
                                                  (True, True, False), (True, True, True)):
            training = build_cases(augmented=augmented, malformed=malformed, contrastive=contrastive)
            held = build_cases(True, augmented=augmented, malformed=malformed, contrastive=contrastive)
            self.assertFalse({row['question'] for row in training} & {row['question'] for row in held})
            self.assertEqual({row['label'] for row in training}, {0, 1})
        gate = IntentGate(3)
        features = torch.randn(2, 2, 3, 3)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        original = gate(features, mask)
        changed = features.clone()
        changed[~mask[:, None, :, None].expand_as(changed)] = 1000
        self.assertTrue(torch.allclose(original, gate(changed, mask)))
        training = {row['question'] for row in build_cases(augmented=True, malformed=True, contrastive=True)}
        self.assertFalse(training & {row[2] for row in SECOND_CASES})

    def test_number_invariant_intent_data_has_no_normalized_overlap(self):
        self.assertEqual(hide_numbers('对象26的值是多少？'), hide_numbers('对象8的值是多少？'))
        training = [row for row in build_cases(augmented=True, malformed=True, contrastive=True)
                    if row['category'] not in ('out_of_range', 'malformed')]
        held = build_cases(True, augmented=True, malformed=True, contrastive=True)
        train_unique = unique_semantic_cases(training)
        held_unique = unique_semantic_cases([row for row in held if valid_finite_question_form(row['question'])],
                                            enforce_labels=False)
        self.assertFalse({r['question'] for r in train_unique} & {r['question'] for r in held_unique})
        self.assertFalse({r['question'] for r in train_unique} & {hide_numbers(r[2]) for r in SECOND_CASES})

    def test_initial_routing_fails_closed_on_invalid_inputs(self):
        self.assertEqual(initial_route(True, 0.7, [12, 12]), 'inspect')
        for form, score, predicted in ((False, 0.9, [1, 1]), (True, 0.2, [1, 1]),
                                       (True, float('nan'), [1, 1]), (True, 0.9, [1, 2]),
                                       (True, 0.9, [-1, -1]), (True, 0.9, [True, True])):
            self.assertEqual(initial_route(form, score, predicted), 'defer')

    def test_multitype_gate_outputs_one_logit_per_type(self):
        gate = MultiIntentGate(3)
        features = torch.randn(2, 2, 3, 3)
        output = gate(features, torch.tensor([[True, True, False], [True, False, False]]))
        self.assertEqual(tuple(output.shape), (2, len(CLASSES)))
        self.assertEqual(checkpoint_payload(gate, (24, 28))['hidden_state_indices'], (24, 28))
        with self.assertRaises(ValueError):
            checkpoint_payload(gate, torch.tensor([24, 28]))
        with self.assertRaises(ValueError):
            unique_cases([{'question': '对象8的值是多少？', 'category': 'current'},
                          {'question': '对象9的值是多少？', 'category': 'history'}])

    def test_hardening_comparison_counts_regression_and_rejects_missing_scores(self):
        rows = [{'question': '对象8的值是多少？', 'label': 1},
                {'question': '不要读取对象9的值。', 'label': 0}]
        result = evaluate_hardened_gate(rows, [0.8, 0.1], [0.9, 0.9])
        self.assertEqual(result['summary']['old']['correct'], 2)
        self.assertEqual(result['summary']['new']['false_accept'], 1)
        with self.assertRaises(ValueError):
            evaluate_hardened_gate(rows, [0.8], [0.9, 0.9])
