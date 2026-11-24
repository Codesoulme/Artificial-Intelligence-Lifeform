"""Compare date and revision coordinates over a single addressed memory timeline."""

from dataclasses import dataclass
from datetime import date
import re
import unicodedata

from .calendar_history import _DATE, _KNOWLEDGE, _date


_DATES = re.compile(_DATE)
_LATEST = re.compile(r'目前|当前|现行|现在|此刻|当下|最新')
_FIRST = re.compile(r'最早|首版|初版|头一[版份]|起初')
_ORDINAL = re.compile(r'第([1-9一二三四五六七八九])(?:个版本|版|份)')
_PREVIOUS = re.compile(r'上一?版|上一份|前一版')
_CHANGE = re.compile(r'更正|改动|变更|改版|换版|切换|调整')
_BEFORE = re.compile(r'之前|以前|前夕|前一天|生效前')
_AFTER = re.compile(r'之后|当天|当日|生效时|生效后|刚生效')


@dataclass(frozen=True)
class VersionQuery:
    kind: str
    coordinate: int | date | None = None
    known_by: date | None = None


def parse_version_query(question):
    text = unicodedata.normalize('NFKC', question)
    if re.search(r'不要|别查|别用|不读取|分别', text):
        return None
    cutoffs = list(_KNOWLEDGE.finditer(text))
    if len(cutoffs) > 1:
        return None
    try:
        known_by = _date(cutoffs[0]['date']) if cutoffs else None
        if cutoffs:
            a, b = cutoffs[0].span()
            text = text[:a] + text[b:]
        days = [_date(match[0]) for match in _DATES.finditer(text)]
    except ValueError:
        return None
    if len(days) == 1:
        before, after = bool(_BEFORE.search(text)), bool(_AFTER.search(text))
        if before and after:
            return None
        if _CHANGE.search(text) and (before or after):
            return VersionQuery('before_change' if before else 'after_change', days[0], known_by)
        if before or (after and not re.search(r'当天|当日', text)):
            return None
        return VersionQuery('as_of', days[0], known_by)
    if days:
        return None
    ordinals = [int(digit) if digit.isascii() else '一二三四五六七八九'.index(digit) + 1
                for digit in _ORDINAL.findall(text)]
    choices = []
    if _LATEST.search(text):
        choices.append(VersionQuery('current', known_by=known_by))
    if _FIRST.search(text):
        choices.append(VersionQuery('ordinal', 1, known_by))
    if _PREVIOUS.search(text):
        choices.append(VersionQuery('previous', known_by=known_by))
    choices.extend(VersionQuery('ordinal', number, known_by) for number in ordinals)
    return choices[0] if len(choices) == 1 else None


def effective_order(revisions, *, as_of, known_by):
    """Return observed slots in effective order, replacing same-date corrections."""
    by_effective = {}
    for slot, (effective, observed) in enumerate(revisions):
        if effective <= as_of and observed <= known_by:
            existing = by_effective.get(effective)
            if existing is None or (observed, slot) > (revisions[existing][1], existing):
                by_effective[effective] = slot
    return tuple(by_effective[day] for day in sorted(by_effective))


def select_version(revisions, request, today):
    if request is None:
        return None
    known_by = request.known_by if request.known_by is not None else today
    if not revisions or known_by > today:
        return None
    dated = request.kind in ('as_of', 'before_change', 'after_change')
    as_of = request.coordinate if dated else today
    if as_of > today:
        return None
    candidates = effective_order(revisions, as_of=as_of, known_by=known_by)
    if request.kind in ('before_change', 'after_change'):
        if len(candidates) < 2 or revisions[candidates[-1]][0] != as_of:
            return None
        return candidates[-2 if request.kind == 'before_change' else -1]
    if request.kind in ('current', 'as_of'):
        return candidates[-1] if candidates else None
    if request.kind == 'previous':
        return candidates[-2] if len(candidates) >= 2 else None
    if request.kind == 'ordinal':
        index = request.coordinate - 1
        return candidates[index] if 0 <= index < len(candidates) else None
    raise ValueError(f'Unsupported version request {request.kind}')
