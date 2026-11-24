"""Conservative routing for a single recorded subject, field and version."""

import re
import unicodedata

from .research_version_timeline import VersionQuery, parse_version_query, select_version
from .temporal_routing import CURRENT


_FIELD_LEAD = re.compile(r'(?:^|[，。；：、\s]|(?:核对|查询|请查|看看|说说|回忆|告诉我)(?:一下)?)$')
_FIELD_TAIL = re.compile(r'(?:$|[，。；：、？?\s]|目前|现在|当时|当日|生效|记录|登记|写|填|还|在|是|的记录)')
_COLLOQUIAL_FIELD_TAIL = re.compile(r'记的(?:是|有)')
_LOOKUP = re.compile(r'哪[一份项个条种款本]|哪个|什[么麼]|啥|告诉我|说说|只报|查一下|核对')
_NONREAD = re.compile(r'翻译|润色|改写|造句|(?:(?:不要|不必|无需|不用|请勿|别)(?:再)?(?:查|读取|读|翻))|分别')
_DATE_HINT = re.compile(r'\d{4}\s*[-/\.年]|[0-9一二三四五六七八九十]+\s*月')
_ORDINAL_HINT = re.compile(r'第\s*[0-9一二三四五六七八九十几]')
_RELATIVE_HINT = re.compile(r'上[一回次版]|前一|首版|初版|最早|起初|当初|那回|当时|旧版|原版'
                            r'|之前|以后|以前|之后|生效[前后]|更正|变更|切换|改动')


def addressed_subject(question, fingerprints, fields_by_key, *, colloquial=False):
    """Require a single enrolled mention with a bounded possessive field."""
    if not question or len(question) > 512:
        return None
    spans = fingerprints.matching_spans(question, max_name_chars=4)
    if len({key for _, _, key in spans}) != 1:
        return None
    def field_matches(end, label):
        suffix = question[end + 1 + len(label):]
        return question.startswith('的' + label, end) and (
            _FIELD_TAIL.match(suffix) or (colloquial and _COLLOQUIAL_FIELD_TAIL.match(suffix)))

    valid = {key for start, end, key in spans if _FIELD_LEAD.search(question[:start]) and any(
        field_matches(end, label) for label in fields_by_key.get(key, ()))}
    return next(iter(valid)) if len(valid) == 1 else None


def choose_scoped_version(question, revisions, today, learned_label):
    """Fall back to the current class only when no explicit scope was lost."""
    text = unicodedata.normalize('NFKC', question)
    if not _LOOKUP.search(text) or _NONREAD.search(text):
        return None, 'outside_lookup_scope'
    request = parse_version_query(text)
    if request is not None:
        if (_DATE_HINT.search(text) and request.known_by is None
                and request.kind not in ('as_of', 'before_change', 'after_change')):
            return None, 'unresolved_time'
        if _ORDINAL_HINT.search(text) and request.kind != 'ordinal':
            return None, 'unresolved_time'
        slot = select_version(revisions, request, today)
        return slot, 'timeline' if slot is not None else 'no_matching_version'
    if learned_label != CURRENT or any(pattern.search(text) for pattern in (_DATE_HINT, _ORDINAL_HINT, _RELATIVE_HINT)):
        return None, 'unresolved_time'
    slot = select_version(revisions, VersionQuery('current'), today)
    return slot, 'learned_current' if slot is not None else 'no_matching_version'
