"""Recover explicitly delimited enrolled names around a numbered correction."""

import re

from .explicit_span_address import explicit_span_address
from .revision_routing import parse_revision_reference


_LEADING = re.compile(r'\s*(?:请查一下|请查|请问一下|请问|麻烦查一下|麻烦查|我想知道|想知道|核对一下|核对|查一下)?\s*')
_FOLLOWING = re.compile(r'(?:\s*[,，：:、]\s*(?:给|为|替)?\s*|\s*(?:给|为|替)\s*)'
                        r'(?P<name>[\u3400-\u9fff]{1,12}?)'
                        r'(?=的(?:饮品|偏好|记录|资料|选择)|登记|记录|被记|采用|喜欢|选|用了)')
_CLAUSE = re.compile(r'[，,。；;：:！？!?\n]')
_NAME = re.compile(r'[\u3400-\u9fff]{1,12}')
_JOIN = re.compile(r'[和与及、]')


def revision_subject(fingerprints, question):
    """Resolve one whole, syntactically delimited span using the enrolled fingerprints."""
    anchor = parse_revision_reference(question, include_span=True)
    if anchor is None or anchor['status'] != 'anchored':
        return {'key': None, 'reason': 'no_single_revision_subject'}
    start, stop = anchor['span']
    prefix = question[:start]
    previous_clauses = list(_CLAUSE.finditer(prefix))
    prefix_start = previous_clauses[-1].end() if previous_clauses else 0
    leading = _LEADING.match(prefix[prefix_start:])
    name_start = prefix_start + leading.end()
    preceding = question[name_start:start].rstrip()
    if preceding.endswith(('在', '的')):
        preceding = preceding[:-1].rstrip()
    following = _FOLLOWING.match(question[stop:])
    spans = []
    if _NAME.fullmatch(preceding):
        spans.append((name_start, name_start + len(preceding)))
    if following is not None:
        spans.append((stop + following.start('name'), stop + following.end('name')))
    if not spans:
        return {'key': None, 'reason': 'no_delimited_revision_subject'}
    names = [question[left:right] for left, right in spans]
    if len(set(names)) != 1:
        return {'key': None, 'reason': 'ambiguous_revision_subject'}
    result = explicit_span_address(fingerprints, question, spans[0], names[0])
    if result['key'] is None and _JOIN.search(names[0]):
        return {'key': None, 'reason': 'ambiguous_revision_subject'}
    return {'name': names[0], 'key': result['key'], 'reason': result['reason']}
