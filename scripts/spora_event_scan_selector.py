"""Experimental chronological selection over neural provenance; not a safety gate."""
from spora.internal_memory import read_provenance_batch


def select_event(model, state, key, order, as_of=None, max_scan=4096, batch_size=16):
    """Find an observation or effective-time candidate.

    Observation order is not effective time. A neural unknown may hide an event;
    even a returned candidate cannot certify completeness or correctness.
    """
    if order not in ('first', 'latest', 'previous', 'effective_first', 'effective_latest', 'effective_previous'):
        raise ValueError('Unsupported event order')
    if state.internal is None or type(key) is not int or not 0 <= key < model.config.key_count:
        raise ValueError('Valid key and provenance required')
    if type(max_scan) is not int or max_scan < 1:
        raise ValueError('Positive scan budget required')
    effective = order.startswith('effective_')
    if effective and (type(as_of) is not int or as_of < 0):
        raise ValueError('Effective-time selection requires a nonnegative as-of time')
    if not effective and as_of is not None:
        raise ValueError('As-of time only applies to effective-time selection')
    if state.internal.start_cursor != 0:
        return {'status': 'abstain', 'reason': 'incomplete_provenance_history'}
    if state.cursor > max_scan:
        return {'status': 'abstain', 'reason': 'scan_budget_exceeded'}
    rows = read_provenance_batch(model, state, [(key, event) for event in range(state.cursor)], batch_size)
    candidates = []
    effective_candidates = {}
    for event, row in enumerate(rows):
        if row.get('reason') == 'unknown_event':
            continue
        if 'reason' in row:
            return {'status': 'abstain', 'reason': 'uncertain_provenance', 'event_time': event}
        if row.get('action') in ('new', 'correct'):
            candidates.append(event)
            if effective:
                valid_at = row.get('valid_at')
                if type(valid_at) is not int or valid_at < 0:
                    return {'status': 'abstain', 'reason': 'missing_effective_time', 'event_time': event}
                if valid_at <= as_of:
                    effective_candidates[valid_at] = max(event, effective_candidates.get(valid_at, -1))
    if effective:
        times = sorted(effective_candidates)
        index = {'effective_first': 0, 'effective_latest': -1, 'effective_previous': -2}[order]
        if len(times) < (2 if order == 'effective_previous' else 1):
            return {'status': 'abstain', 'reason': 'insufficient_effective_candidates'}
        effective_time = times[index]
        return {'status': 'experimental_candidate', 'event_time': effective_candidates[effective_time],
                'effective_time': effective_time, 'candidate_count': len(times),
                'checked_events': state.cursor,
                'warning': 'neural_unknown_may_hide_events;effective_time_decoded_from_provenance'}
    index = {'first': 0, 'latest': -1, 'previous': -2}[order]
    if len(candidates) < (2 if order == 'previous' else 1):
        return {'status': 'abstain', 'reason': 'insufficient_observed_candidates'}
    return {'status': 'experimental_candidate', 'event_time': candidates[index],
            'candidate_count': len(candidates), 'checked_events': state.cursor,
            'warning': 'neural_unknown_may_hide_events;observation_order_only'}
