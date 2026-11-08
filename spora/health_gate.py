"""Sealed-evaluation promotion gate inspired by EvalRX."""
from __future__ import annotations
import hashlib
import json


def cases_digest(cases):
    return hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def promotion_decision(baseline, candidate, *, expected_cases_digest,
                       candidate_cases_digest, lower_is_better=() , max_regressions=0):
    if candidate_cases_digest != expected_cases_digest:
        return {'promote': False, 'reason': 'sealed_cases_changed'}
    regressions = []
    for name in baseline:
        if name not in candidate:
            return {'promote': False, 'reason': 'metric_missing:'+name}
        if name in lower_is_better and candidate[name] > baseline[name]:
            regressions.append(name)
        elif name not in lower_is_better and candidate[name] < baseline[name]:
            regressions.append(name)
    if len(regressions) > max_regressions:
        return {'promote': False, 'reason': 'regression', 'regressions': regressions}
    return {'promote': True, 'reason': 'sealed_metrics_passed', 'regressions': regressions}
