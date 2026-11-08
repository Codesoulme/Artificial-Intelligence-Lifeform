import torch
from torch import nn
from torch.nn import functional as F

from scripts.benchmark_p112_microbatch import split_batches
from spora.token_objective import backward_token_budget


def test_split_and_token_weighted_gradient():
    torch.manual_seed(112)
    model = nn.Embedding(11, 11)
    x = torch.tensor([[1, 2, 3, 0], [2, 3, 4, 5], [3, 4, 0, 0], [4, 5, 6, 0], [5, 6, 7, 8]])
    y = torch.tensor([[2, 3, -100, -100], [3, 4, 5, 6], [4, -100, -100, -100],
                      [5, 6, 7, -100], [6, 7, 8, 9]])

    class Model(nn.Module):
        def forward(self, tokens, state):
            return model(tokens)

    wrapped = Model()
    wrapped.add_module('embedding', model)
    F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100).backward()
    expected = model.weight.grad.clone()
    for size in (1, 2, 4, 5):
        model.zero_grad(set_to_none=True)
        pieces = split_batches(x, y, size)
        _, count = backward_token_budget(wrapped, None, pieces)
        assert count == 14
        torch.testing.assert_close(model.weight.grad, expected, atol=1e-7, rtol=1e-6)
