"""Deterministic allocation by observed supervised tokens, not batch counts."""
import math


def choose_source(shares, counts):
    if not shares or any(k not in ('pretrain','teacher','dialogue','foundation') for k in shares):
        raise ValueError('Unknown or empty source mix')
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in shares.values()) or abs(sum(shares.values())-1)>1e-6:
        raise ValueError('Positive source shares summing to one required')
    observed={k:counts.get(k+'_source_targets',0) for k in shares}
    if any(type(v) is not int or v<0 for v in observed.values()):
        raise ValueError('Nonnegative integer token counts required')
    # A batch is indivisible. Future lengths are unknown, so realized shares can
    # overshoot by a batch; measure actual totals rather than claiming exactness.
    return min(shares,key=lambda kind:observed[kind]/shares[kind])
