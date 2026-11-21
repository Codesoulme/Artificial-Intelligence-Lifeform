"""Resolve explicit correction ordinals against indexed neural history."""

import re
import unicodedata

import torch

from .internal_memory import read_provenance_batch


_ORDINAL = re.compile(r'第([0-9０-９一二两三四五六七八九十零〇百千]+)次\s*(?:更正|修改|修订|更新|调整)')
_BEFORE = re.compile(r'\s*(?:发生)?(?:之前|以前|前)')
_AFTER = re.compile(r'\s*(?:完成后|生效后|之后|以后|后)')
_BOTH = re.compile(r'\s*(?:发生)?(?:前后|前(?:和|与|及|、)后)')
_DIGITS = {char: number for number, char in enumerate('一二三四五六七八九', start=1)}
_DIGITS['两'] = 2


def _ordinal_number(text):
    if text.isascii() and text.isdigit():
        return int(text) if len(text) <= 3 else None
    if text in _DIGITS:
        return _DIGITS[text]
    left, ten, right = text.partition('十')
    if ten and (not left or left in _DIGITS) and (not right or right in _DIGITS):
        return 10 * _DIGITS.get(left, 1) + _DIGITS.get(right, 0)
    return None


def parse_revision_reference(question, *, include_span=False):
    """Return an explicit ordinal and side, or abstain on an ambiguous reference."""
    if not isinstance(question, str):
        raise ValueError('Expected a question string')
    anchors = list(_ORDINAL.finditer(question))
    if not anchors:
        return None
    if len(anchors) != 1:
        return {'status': 'abstain', 'reason': 'multiple_correction_anchors'}
    anchor = anchors[0]
    for left, right in (('「', '」'), ('“', '”'), ('『', '』')):
        if question.rfind(left, 0, anchor.start()) > question.rfind(right, 0, anchor.start()) and (
            question.find(right, anchor.end()) >= 0
        ):
            return {'status': 'abstain', 'reason': 'quoted_correction_reference'}
    number = _ordinal_number(unicodedata.normalize('NFKC', anchor[1]))
    if number is None or number < 1:
        return {'status': 'abstain', 'reason': 'invalid_correction_ordinal'}
    suffix = question[anchor.end():]
    if _BOTH.match(suffix):
        return {'status': 'abstain', 'reason': 'ambiguous_correction_side'}
    direction = _BEFORE.match(suffix)
    if direction is not None:
        side = 'before'
    elif (direction := _AFTER.match(suffix)) is not None:
        side = 'after'
    else:
        return {'status': 'abstain', 'reason': 'missing_correction_side'}
    result = {'status': 'anchored', 'number': number, 'side': side}
    if include_span:
        result['span'] = anchor.start(), anchor.end() + direction.end()
    return result


def select_numbered_revision(model, state, key, number, side, *, max_scan=4096):
    """Select a value event immediately before or after an observed correction."""
    if (type(key) is not int or not 0 <= key < model.config.key_count or type(number) is not int
            or number < 1 or side not in ('before', 'after') or type(max_scan) is not int or max_scan < 1):
        raise ValueError('Invalid correction reference')
    internal = state.internal
    if internal is None or internal.start_cursor != 0 or internal.event_keys is None:
        return {'status': 'abstain', 'reason': 'complete_event_key_index_required'}
    internal.validate(model, state)
    index = torch.cat(internal.event_keys)[:state.cursor]
    events = (index == key).nonzero(as_tuple=True)[0].tolist()
    if len(events) > max_scan:
        return {'status': 'abstain', 'reason': 'scan_budget_exceeded'}
    provenance = read_provenance_batch(model, state, [(key, event) for event in events])
    previous = None
    corrections = 0
    for event, row in zip(events, provenance, strict=True):
        if 'reason' in row:
            return {'status': 'abstain', 'reason': 'unreadable_indexed_provenance', 'event_time': event}
        action = row.get('action')
        if action == 'new' and previous is None:
            previous = event
        elif action == 'correct' and previous is not None:
            corrections += 1
            if corrections == number:
                return {'status': 'experimental_candidate', 'event_time': previous if side == 'before' else event,
                        'correction_event': event, 'checked_events': len(events)}
            previous = event
        elif action not in ('support', 'repeat', 'defer'):
            return {'status': 'abstain', 'reason': 'inconsistent_committed_history', 'event_time': event}
    return {'status': 'abstain', 'reason': 'correction_ordinal_not_found', 'checked_events': len(events)}
