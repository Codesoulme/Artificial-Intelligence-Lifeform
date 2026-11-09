import unittest

import torch

from scripts.open_base_addressed_full_answer import answer_batch


class Tokenizer:
    eos_token_id = 9
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return 'prompt'

    def __call__(self, text, return_tensors):
        class Batch(dict):
            def to(self, device):
                return self
        return Batch(input_ids=torch.tensor([[7, 8]]))

    def encode(self, text, add_special_tokens):
        return [int(c) + 1 for c in text]


class AnswerBatchTests(unittest.TestCase):
    def test_labels_cover_complete_answers_and_end_only(self):
        inputs, labels = answer_batch(Tokenizer(), 'question', [3, 15], 'cpu')
        self.assertEqual(inputs['input_ids'].tolist(), [[7, 8, 4, 9, 0], [7, 8, 2, 6, 9]])
        self.assertEqual(inputs['attention_mask'].tolist(), [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
        self.assertEqual(labels.tolist(), [[-100, -100, 4, 9, -100], [-100, -100, 2, 6, 9]])

    def test_missing_end_token_fails(self):
        tokenizer = Tokenizer()
        tokenizer.eos_token_id = None
        with self.assertRaises(ValueError):
            answer_batch(tokenizer, 'question', [1], 'cpu')
