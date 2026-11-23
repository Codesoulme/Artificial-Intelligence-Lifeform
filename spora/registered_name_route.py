"""Resolve named memory addresses against the persistent individual's fingerprints."""

from .explicit_span_address import explicit_span_address
from .identity_fingerprints import IdentityFingerprints


def resolve_confirmed_name(fingerprints, full_name):
    """Resolve a full name supplied as an explicit subject selection."""
    if fingerprints is None:
        return {'key': None, 'reason': 'no_registered_names'}
    if not isinstance(fingerprints, IdentityFingerprints):
        raise TypeError('Expected the individual identity fingerprints')
    try:
        key = fingerprints.lookup(full_name)
    except ValueError:
        return {'key': None, 'reason': 'invalid_confirmed_name'}
    return {'key': key, 'reason': 'confirmed_enrollment' if key is not None else 'name_not_enrolled'}


def verify_detected_name(fingerprints, question, spans):
    """Apply the research read rule to a detector's complete mention proposals."""
    if not isinstance(question, str) or not question.strip() or len(question) > 512:
        return {'key': None, 'reason': 'invalid_question', 'name': None}
    if not isinstance(spans, (list, tuple)) or len(spans) != 1:
        return {'key': None, 'reason': 'no_single_name', 'name': None}
    if fingerprints is None:
        return {'key': None, 'reason': 'no_registered_names', 'name': None}
    if not isinstance(spans[0], (list, tuple)) or len(spans[0]) != 2:
        return {'key': None, 'reason': 'invalid_span', 'name': None}
    start, end = spans[0]
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(question):
        return {'key': None, 'reason': 'invalid_span', 'name': None}
    name = question[start:end]
    exact = explicit_span_address(fingerprints, question, (start, end), name)
    if exact['key'] is None:
        return {'key': None, 'reason': 'name_not_unique_enrollment', 'name': name}
    hits = fingerprints.matching_spans(question, max_name_chars=12)
    key = exact['key'] if hits == [(start, end, exact['key'])] else None
    return {'key': key, 'reason': 'enrolled_span' if key is not None else 'name_not_unique_enrollment', 'name': name}
