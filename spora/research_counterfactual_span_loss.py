"""Multi-token counterfactual loss for isolated memory-reading experiments.

Each candidate must be evaluated with its own teacher-forced prefix. Callers
slice logits to the relevant value-token positions, and must not pass answer
text or target token IDs into the student question or inference-time output.
"""
import torch
from torch.nn import functional as F


def span_mean_log_probability(logits, targets):
    """Mean log probability of each non-padded target token per example."""
    if logits.ndim != 3 or targets.ndim != 2 or logits.shape[:2] != targets.shape:
        raise ValueError('Expected logits [batch, time, vocabulary] and targets [batch, time]')
    if targets.dtype != torch.long or logits.shape[0] < 1 or logits.shape[1] < 1:
        raise ValueError('Invalid targets or empty batch')
    mask = targets.ne(-100)
    if not bool(mask.any(dim=1).all()):
        raise ValueError('Every value must contain at least one token')
    if bool(((targets[mask] < 0) | (targets[mask] >= logits.shape[-1])).any()):
        raise ValueError('Target token outside vocabulary')
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('Nonfinite logits')
    selected = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    return (selected * mask).sum(dim=-1) / mask.sum(dim=-1)


def counterfactual_span_loss(a_on_a, b_on_a, a_on_b, b_on_b, targets_a, targets_b, margin=1.0):
    """Prefer A under state A and B under state B, including every value token.

    Names specify candidate and state, e.g. b_on_a is the B prefix run while
    state A is active. Each tensor may have a different time length for A and B.
    Length normalization makes per-token scores comparable across unequal spans.
    This loss alone does not ensure generalization, attribution or abstention.
    """
    if not isinstance(margin, (int, float)) or isinstance(margin, bool) or not 0 <= margin < float('inf'):
        raise ValueError('Invalid margin')
    a_a = span_mean_log_probability(a_on_a, targets_a)
    b_a = span_mean_log_probability(b_on_a, targets_b)
    a_b = span_mean_log_probability(a_on_b, targets_a)
    b_b = span_mean_log_probability(b_on_b, targets_b)
    if not (a_a.shape == b_a.shape == a_b.shape == b_b.shape):
        raise ValueError('Counterfactual batches must match')
    return 0.5 * (F.softplus(margin - (a_a - b_a)).mean()
                  + F.softplus(margin - (b_b - a_b)).mean())


def diagnostic_gradient_norms(loss, groups):
    """Measure one loss component's gradients without changing .grad fields."""
    if loss.ndim != 0 or not loss.requires_grad or not groups:
        raise ValueError('Expected differentiable scalar and parameter groups')
    ordered = [(name, list(parameters)) for name, parameters in groups.items()]
    parameters = [p for _, group in ordered for p in group]
    if not parameters or len({id(p) for p in parameters}) != len(parameters):
        raise ValueError('Parameter groups must be nonempty overall and disjoint')
    if any(not p.requires_grad for p in parameters):
        raise ValueError('Diagnostic parameters must require gradients')
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    output, index = {}, 0
    for name, group in ordered:
        selected = gradients[index:index + len(group)]
        index += len(group)
        finite = all(g is None or bool(torch.isfinite(g).all()) for g in selected)
        if not finite:
            raise ValueError('Nonfinite diagnostic gradient')
        square = sum(float(g.detach().float().square().sum()) for g in selected if g is not None)
        output[name] = {'parameters': len(group), 'connected': sum(g is not None for g in selected),
                        'nonzero': sum(g is not None and bool((g != 0).any()) for g in selected),
                        'norm': square ** 0.5}
    return output
