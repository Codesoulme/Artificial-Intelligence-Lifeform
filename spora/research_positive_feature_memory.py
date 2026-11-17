"""Normalized positive-feature attention statistics for isolated research."""
import torch
import torch.nn.functional as F


def features(x):
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or not bool(torch.isfinite(x).all()):
        raise ValueError('Expected a nonempty finite matrix')
    return F.elu(F.normalize(x.float(), dim=-1) * 2.0) + 1.0


def write(keys, values):
    if values.ndim != 2 or values.shape[0] != keys.shape[0] or not bool(torch.isfinite(values).all()):
        raise ValueError('Invalid value matrix')
    phi = features(keys)
    return phi.T @ values.float(), phi.sum(0)


def read(queries, state):
    phi = features(queries)
    matrix, normalizer = state
    if (matrix.ndim != 2 or normalizer.ndim != 1 or matrix.shape[0] != phi.shape[1]
            or normalizer.shape[0] != phi.shape[1]):
        raise ValueError('Invalid memory shape')
    denom = phi @ normalizer
    if not bool(torch.isfinite(denom).all()) or bool((denom <= 0).any()):
        raise ValueError('Empty or invalid memory')
    return (phi @ matrix) / denom.unsqueeze(-1)
