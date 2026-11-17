"""Exact reference for partitioning attention; not a deployed memory reader."""
import torch


def partitioned_attention(query, keys, values, history_mask, local_mask):
    """Merge disjoint history/local attention with exact softmax mass.

    Inputs are [batch, query, width], [batch, key, width], and boolean masks
    [batch, query, key]. Each query must have at least one visible key.
    Returns merged output, historical mass, historical and local outputs.
    """
    if query.ndim != 3 or keys.ndim != 3 or values.ndim != 3:
        raise ValueError('Expected three-dimensional tensors')
    if (query.shape[0] != keys.shape[0] or keys.shape[:2] != values.shape[:2]
            or query.shape[-1] != keys.shape[-1] or query.shape[-1] == 0):
        raise ValueError('Incompatible attention shapes')
    shape = (query.shape[0], query.shape[1], keys.shape[1])
    if (history_mask.shape != shape or local_mask.shape != shape
            or history_mask.dtype != torch.bool or local_mask.dtype != torch.bool
            or bool((history_mask & local_mask).any())
            or not bool((history_mask | local_mask).any(-1).all())):
        raise ValueError('Invalid or empty attention partition')
    logits = query.float() @ keys.float().transpose(-1, -2) / (query.shape[-1] ** 0.5)
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(values).all()):
        raise ValueError('Nonfinite input')

    def part(mask):
        valid = mask.any(-1, keepdim=True)
        masked = logits.masked_fill(~mask, -torch.inf)
        # Softmax of an empty set is undefined; use zero output and -inf mass.
        safe = torch.where(valid, masked, torch.zeros_like(masked))
        weights = torch.softmax(safe, -1) * mask
        return weights @ values.float(), torch.logsumexp(masked, -1, keepdim=True)

    old, old_lse = part(history_mask)
    local, local_lse = part(local_mask)
    mass = torch.exp(old_lse - torch.logaddexp(old_lse, local_lse))
    return mass * old + (1.0 - mass) * local, mass, old, local
