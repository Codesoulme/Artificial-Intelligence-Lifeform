"""Research-only positive random features for an exponential attention kernel."""
import torch


def log_features(x, projection):
    if (x.ndim != 2 or projection.ndim != 2 or x.shape[1] != projection.shape[1]
            or not x.shape[0] or not projection.shape[0]
            or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(projection).all())):
        raise ValueError('Expected finite input and projection matrices')
    scaled = x.float() / (x.shape[1] ** 0.25)
    return scaled @ projection.float().T - 0.5 * scaled.square().sum(-1, keepdim=True)


def write(keys, values, projection):
    if values.ndim != 2 or values.shape[0] != keys.shape[0] or not bool(torch.isfinite(values).all()):
        raise ValueError('Invalid values')
    logs = log_features(keys, projection)
    shift = logs.max()
    phi = (logs - shift).exp()
    return phi.T @ values.float(), phi.sum(0), shift


def merge(first, second):
    """Merge independently scaled states made with the *same* projection.

    The caller must ensure identical projections and token semantics; shapes
    alone cannot establish identity. Inputs are not modified.
    """
    a, az, ashift = first
    b, bz, bshift = second
    if (a.ndim != 2 or a.shape != b.shape or az.shape != bz.shape
            or az.shape != (a.shape[0],) or ashift.ndim != 0 or bshift.ndim != 0
            or any(not bool(torch.isfinite(x).all()) for x in (a, az, ashift, b, bz, bshift))
            or bool((az < 0).any()) or bool((bz < 0).any())):
        raise ValueError('Incompatible or nonfinite memory states')
    shift = torch.maximum(ashift, bshift)
    aw, bw = (ashift - shift).exp(), (bshift - shift).exp()
    return aw * a + bw * b, aw * az + bw * bz, shift


def read(queries, projection, state):
    matrix, total, _ = state
    logs = log_features(queries, projection)
    phi = (logs - logs.max(-1, keepdim=True).values).exp()
    if matrix.shape[0] != phi.shape[1] or total.shape != (phi.shape[1],):
        raise ValueError('Memory dimensions do not match features')
    denom = phi @ total
    if not bool(torch.isfinite(denom).all()) or bool((denom <= 0).any()):
        raise ValueError('Degenerate numerical memory')
    return (phi @ matrix) / denom.unsqueeze(-1)
