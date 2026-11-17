"""Research metrics for missing-evidence decisions; no runtime routing policy."""
import math


def confusion(scores, labels, threshold):
    if len(scores) != len(labels) or not scores:
        raise ValueError('Nonempty aligned scores and labels required')
    if not math.isfinite(threshold) or any(not math.isfinite(s) for s in scores):
        raise ValueError('Finite scores and threshold required')
    if any(y not in (0, 1) for y in labels):
        raise ValueError('Binary labels required')
    result = dict(tp=0, fn=0, fp=0, tn=0)
    for s, y in zip(scores, labels):
        result[('tp' if s >= threshold else 'fn') if y else
               ('fp' if s >= threshold else 'tn')] += 1
    pos, neg = result['tp'] + result['fn'], result['fp'] + result['tn']
    if not pos or not neg:
        raise ValueError('Both classes required')
    result['balanced_error'] = (result['fn'] / pos + result['fp'] / neg) / 2
    return result


def calibrate(scores, labels):
    """Fit only on calibration data; tie break toward fewer missed needs."""
    confusion(scores, labels, 0.0)
    candidates = sorted(set(scores)) + [math.nextafter(max(scores), math.inf)]
    scored = [(confusion(scores, labels, t), t) for t in candidates]
    best, threshold = min(scored, key=lambda x: (x[0]['balanced_error'], x[0]['fn'], x[1]))
    zero_fp = math.nextafter(max(s for s, y in zip(scores, labels) if y == 0), math.inf)
    return {'balanced': threshold, 'zero_calibration_fp': zero_fp}, best


def rank_auc(scores, labels):
    confusion(scores, labels, 0.0)
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
