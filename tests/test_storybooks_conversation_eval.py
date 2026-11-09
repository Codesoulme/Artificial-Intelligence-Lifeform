import math
import unittest

import torch

from scripts.evaluate_storybooks_conversation import text_nll


class ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.inputs = []

    def forward(self, tokens, state):
        self.inputs.append(tokens.tolist()[0])
        return torch.zeros((*tokens.shape, 10), device=tokens.device) + self.anchor


class StorybookConversationEvalTests(unittest.TestCase):
    def test_each_content_target_counted_once_with_distinct_bos(self):
        model = ConstantModel()
        loss = text_nll(model, None, [2, 3, 4, 5, 6], bos=1, length=4)
        self.assertAlmostEqual(loss, 5 * math.log(10), places=5)
        self.assertEqual(model.inputs, [[1, 2, 3, 4], [4, 5, 6]])

    def test_empty_content_rejected(self):
        with self.assertRaises(ValueError):
            text_nll(ConstantModel(), None, [], bos=1, length=4)


if __name__ == '__main__':
    unittest.main()
