"""Token-weighted microbatch backpropagation for limited device memory."""
import torch
from torch.nn import functional as F


def backward_supervised(model, state, tokens, labels, microbatch):
    """Accumulate the same mean supervised-token gradient as a full batch.

    The caller zeroes gradients and performs clipping/one optimizer update.
    Different answer lengths must not give each microbatch equal weight.
    """
    if tokens.shape != labels.shape or tokens.ndim != 2 or microbatch < 1:
        raise ValueError('Matching matrices and positive microbatch required')
    total = int((labels != -100).sum())
    if total == 0:
        raise ValueError('No supervised targets')
    result = torch.zeros((), device=tokens.device)
    for start in range(0, len(tokens), microbatch):
        target = labels[start:start + microbatch]
        if not bool((target != -100).any()):
            continue
        logits = model(tokens[start:start + microbatch], state)
        part = F.cross_entropy(logits.flatten(0, 1), target.flatten(),
                               ignore_index=-100, reduction='sum') / total
        if not bool(torch.isfinite(part)):
            raise ValueError('Nonfinite objective')
        part.backward()
        result += part.detach()
        del logits, part
    return result
