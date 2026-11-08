import unittest

import torch

from scripts.evaluate_context_contrast import score


class DummyTokenizer:
    def encode(self, value):
        class Result:
            ids = [2] if value == "a" else [3]
        return Result()

    def token_to_id(self, token):
        return 4


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, tokens, state):
        logits = torch.zeros((1, tokens.shape[1], 5))
        logits[0, 1, 2] = 8.0
        logits[0, 2, 4] = 8.0
        return logits


class ContrastTests(unittest.TestCase):
    def test_scores_answer_after_prefix_and_termination(self):
        model = DummyModel()
        tokenizer = DummyTokenizer()
        self.assertGreater(score(model, None, tokenizer, [0, 1], "a"),
                           score(model, None, tokenizer, [0, 1], "b"))


if __name__ == "__main__":
    unittest.main()
