"""Counterfactual contrast for a shared, value-free answer prefix."""
import torch
import torch.nn.functional as F


def pair_loss(logits_a, logits_b, target_a, target_b, margin=1.0):
    if (logits_a.ndim != 2 or logits_a.shape != logits_b.shape or logits_a.shape[0] != 1
            or type(target_a) is not int or type(target_b) is not int
            or not 0 <= target_a < logits_a.shape[1] or not 0 <= target_b < logits_a.shape[1]
            or target_a == target_b or not 0 <= margin < float('inf')):
        raise ValueError('Invalid pair comparison')
    if not bool(torch.isfinite(logits_a).all() and torch.isfinite(logits_b).all()):
        raise ValueError('Nonfinite logits')
    m_a=logits_a[:,target_a]-logits_a[:,target_b]
    m_b=logits_b[:,target_b]-logits_b[:,target_a]
    return 0.5*(F.softplus(margin-m_a).mean()+F.softplus(margin-m_b).mean())
