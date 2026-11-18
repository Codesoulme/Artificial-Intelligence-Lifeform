"""Opt-in focus on an actually recorded neural event, without storing its answer.

The two integers are exact routing metadata, not learned memory or proof of
identity/truth. Upstream callers must resolve the object and indicate which
foreground observations update attention. Content and provenance are read from
the individual's existing neural history on every use; there is no fallback to
another event when the chosen one cannot be verified.
"""
from dataclasses import replace

import torch

from .growing import query
from .internal_memory import observe_sourced, read_provenance
from .sleep_memory import history_site


def _internal(model, state, *, enabled=True):
    if state.internal is None or state.internal.event_keys is None:
        raise ValueError('Event focus requires an explicit event-key index')
    state.internal.validate(model, state)
    if enabled and state.internal.event_focus is None:
        raise ValueError('Event focus is not enabled')
    return state.internal


def enable_event_focus(model, state):
    internal = _internal(model, state, enabled=False)
    if internal.event_focus is not None:
        raise ValueError('Event focus is already enabled')
    focus = torch.full((2,), -1, dtype=torch.int64, device=internal.evidence.device)
    return replace(state, internal=replace(internal, event_focus=focus))


def clear_event_focus(model, state):
    internal = _internal(model, state)
    return replace(state, internal=replace(internal, event_focus=torch.full_like(internal.event_focus, -1)))


@torch.no_grad()
def _inspect(model, state, key, event_time):
    if (type(key) is not int or not 0 <= key < model.config.key_count
            or type(event_time) is not int or not 0 <= event_time < state.cursor):
        return {'status': 'abstain', 'reason': 'invalid_event_address'}
    owner = int(state.internal.event_keys[event_time // model.config.time_count][event_time % model.config.time_count])
    if owner != key:
        return {'status': 'abstain', 'reason': 'event_object_mismatch'}
    keys = torch.tensor([key], dtype=torch.long, device=next(model.parameters()).device)
    modes = torch.ones_like(keys)
    _, bound = model.representation(keys, modes, torch.full_like(keys, event_time % model.config.time_count))
    page = state.pages[event_time // model.config.time_count]
    for index, site in enumerate(model.memories):
        address = model.memory_address(site, bound, keys, modes)
        retrieved = site.retrieve([history_site(page, index)[0]], address)
        # A non-unknown classifier output is possible even when the memory
        # contribution is exactly zero. This is only a structural check, not a
        # calibrated confidence or correctness guarantee for nonzero vectors.
        if not bool(torch.isfinite(retrieved).all()) or not bool((retrieved != 0).any()):
            return {'status': 'abstain', 'reason': 'empty_or_invalid_event_read', 'site': index}
    provenance = read_provenance(model, state, key, event_time)
    if ('reason' in provenance or provenance.get('source') not in ('direct', 'report')
            or provenance.get('action') not in ('new', 'correct')
            or type(provenance.get('valid_at')) is not int
            or not 0 <= provenance['valid_at'] <= event_time):
        return {'status': 'abstain', 'reason': 'unverified_factual_event', 'provenance': provenance}
    logits = query(model, state, keys, torch.ones_like(keys), torch.full_like(keys, event_time))
    if (logits.shape != (1, model.config.value_count) or not bool(torch.isfinite(logits).all())
            or int(logits.argmax(-1)[0]) == model.config.unknown):
        return {'status': 'abstain', 'reason': 'focused_event_unreadable'}
    return {'status': 'experimental_candidate', 'key': key, 'event_time': event_time,
            'provenance': provenance}


def bind_event_focus(model, state, key, event_time):
    internal = _internal(model, state)
    result = _inspect(model, state, key, event_time)
    if result['status'] != 'experimental_candidate':
        raise ValueError('Cannot bind event focus: ' + result['reason'])
    focus = torch.tensor([key, event_time], dtype=torch.int64, device=internal.evidence.device)
    return replace(state, internal=replace(internal, event_focus=focus))


def read_focused_event(model, state):
    if state.internal is None or state.internal.event_focus is None:
        return {'status': 'abstain', 'reason': 'event_focus_disabled'}
    internal = _internal(model, state)
    key, event = internal.event_focus.tolist()
    if (key, event) == (-1, -1):
        return {'status': 'abstain', 'reason': 'no_event_focus'}
    return _inspect(model, state, key, event)


def observe_focused(model, state, key, value, source, quality, valid_at=None):
    """Foreground write and focus as one candidate state; callers persist it.

    A failed readback aborts the candidate. Background observation should use
    observe_sourced directly, which preserves the previous focus.
    """
    _internal(model, state)
    candidate, decision = observe_sourced(model, state, key, value, source, quality, valid_at)
    if decision['current_write']:
        candidate = bind_event_focus(model, candidate, key, decision['history_time'])
    return candidate, decision


@torch.no_grad()
def focused_memory_args(model, state):
    """Build decoder controls from the stored event, never from an answer table."""
    selection = read_focused_event(model, state)
    if selection['status'] != 'experimental_candidate':
        raise ValueError('Cannot read event focus: ' + selection['reason'])
    device = next(model.parameters()).device
    key = torch.tensor([selection['key']], dtype=torch.long, device=device)
    modes = torch.ones_like(key)
    local = torch.full_like(key, selection['event_time'] % model.config.time_count)
    _, bound = model.representation(key, modes, local)
    addresses = [model.memory_address(site, bound, key, modes) for site in model.memories]
    return {'memory_addresses': addresses, 'current_only': False,
            'memory_slots': torch.full_like(key, 1 + selection['event_time'] // model.config.time_count),
            'memory_read_mask': torch.ones_like(key, dtype=torch.bool)}, selection
