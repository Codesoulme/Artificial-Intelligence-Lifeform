"""Research-only explicit text selection to an already enrolled memory slot.

The caller supplies both the selected character range and the name. This is
not identity authentication, semantic disambiguation, or a language parser.
"""
from .identity_fingerprints import IdentityFingerprints


def explicit_span_address(fingerprints, question, span, name):
    if not isinstance(fingerprints, IdentityFingerprints):
        raise TypeError('Explicit identity fingerprints required')
    if not isinstance(question, str) or not question.strip():
        raise ValueError('A nonempty question is required')
    if name is None and span is None:
        return {'key': None, 'reason': 'explicit_selection_required'}
    if (not isinstance(name, str) or not name or not isinstance(span, (tuple, list)) or len(span) != 2
            or any(type(i) is not int for i in span)):
        return {'key': None, 'reason': 'invalid_selection'}
    start, end = span
    if not 0 <= start < end <= len(question) or question[start:end] != name:
        return {'key': None, 'reason': 'invalid_selection'}
    try:
        key = fingerprints.lookup(name)
    except ValueError:
        key = None
    return {'key': key, 'reason': 'enrolled' if key is not None else 'explicit_name_not_enrolled'}
