"""Compute vocabulary logits only at supervised positions during training.

Attention and neural memory still process the complete causal input. Dropping
unlabelled vocabulary projections changes neither the objective nor the exact
real-arithmetic gradient. Floating-point kernels can differ slightly.
"""
import torch
from torch.nn import functional as F


def supervised_logits(model, state, tokens, selected):
    if tokens.ndim != 2 or tokens.dtype != torch.long or selected.shape != tokens.shape:
        raise ValueError('Matching token and selection matrices required')
    if selected.dtype != torch.bool or not 0 < tokens.shape[1] <= model.config.context:
        raise ValueError('Boolean selection and valid context required')
    x = model.embedding(tokens)
    site = 0
    for i, block in enumerate(model.blocks):
        x, _ = block(x)
        if i in model.config.memory_layers:
            x = model.read_memory(x, state, site)
            site += 1
    return F.linear(model.norm(x[selected]), model.embedding.weight)


def backward_selected(model, state, tokens, labels, microbatch):
    if tokens.shape != labels.shape or tokens.ndim != 2 or microbatch < 1:
        raise ValueError('Matching matrices and positive microbatch required')
    total = int((labels != -100).sum())
    if total == 0:
        raise ValueError('No supervised targets')
    result = torch.zeros((), device=tokens.device)
    for start in range(0, len(tokens), microbatch):
        target = labels[start:start + microbatch]
        selected = target != -100
        if not bool(selected.any()):
            continue
        logits = supervised_logits(model, state, tokens[start:start + microbatch], selected)
        part = F.cross_entropy(logits, target[selected], reduction='sum') / total
        if not bool(torch.isfinite(part)):
            raise ValueError('Nonfinite objective')
        part.backward()
        result += part.detach()
        del logits, part
    return result
