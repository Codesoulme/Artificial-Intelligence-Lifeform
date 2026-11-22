"""Resolve explicit Chinese date queries against a saved observation calendar."""

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
import re
import unicodedata


_DATE = r'(?:[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日)'
_KNOWLEDGE = re.compile(rf'(?:(?:只)?(?:按|根据|依据))?(?:截至|截止到)(?P<date>{_DATE})'
                        r'(?:为止)?(?:已经|已)?(?:收到|掌握)的(?:记录|资料|信息)')
_PRESENT = re.compile(r'(?:按|根据|依据)(?:现在|目前)掌握的(?:完整)?(?:记录|资料|信息)')
_LEAD = r'(?:请问|请查|麻烦查|帮我查)?\s*'
_NAME = r'(?P<name>[\u3400-\u9fff]{1,12}?)'
_PREDICATE = r'(?:饮品|饮料)(?:是什么|是哪种|是哪一项|填的是什么)\s*[？?。]?'
_CORE = (
    re.compile(rf'{_LEAD}(?P<date>{_DATE})(?:当天|时)?(?:对|给|为){_NAME}'
               rf'(?:有效|登记|记录)的{_PREDICATE}'),
    re.compile(rf'{_LEAD}{_NAME}(?:在|于)(?P<date>{_DATE})(?:当天|时)?'
               rf'(?:的|登记的|记录的|有效的){_PREDICATE}'),
)
_ENDING = re.compile(r'[,，。?]\s*只(?:回答|说|给出)(?:饮品|饮料)(?:名称|名字)?[。！!]?\s*$')
_SEPARATOR = re.compile(r'\s*[,，;；]\s*')
_FOLLOW_PREFIX = re.compile(r'^(?:那么|那|同一个人|还是这个人|还是他|还是她)\s*')
_FOLLOW_TAIL = re.compile(r'(?:呢)?[?？。]?\s*$')
_DATE_ONLY = re.compile(rf'(?P<date>{_DATE})(?:当天|时)?')
_BEFORE_BACKFILL = re.compile(r'(?:按)?补录(?:到来|收到|出现)?(?:之前|以前|前)(?:的(?:记录|资料|信息))?')


def _date(text):
    if '年' not in text:
        return date.fromisoformat(text)
    year, month, day = re.split('[年月日]', text)[:3]
    return date(int(year), int(month), int(day))


@dataclass(frozen=True)
class ObservationCalendar:
    days: tuple[date, ...]

    @classmethod
    def from_payload(cls, payload, cursor):
        if (not isinstance(payload, dict) or payload.keys() != {'format', 'days'}
                or payload['format'] != 'spora-observation-calendar-v1'
                or not isinstance(payload['days'], list) or len(payload['days']) != cursor
                or any(not isinstance(day, str) or not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', day)
                       for day in payload['days'])):
            raise ValueError('Calendar must describe every observation in the saved individual')
        days = tuple(date.fromisoformat(day) for day in payload['days'])
        if any(a > b for a, b in zip(days, days[1:])):
            raise ValueError('Calendar dates must be nondecreasing')
        return cls(days)

    def payload(self):
        return {'format': 'spora-observation-calendar-v1', 'days': [day.isoformat() for day in self.days]}

    def append(self, day):
        if type(day) is not date or (self.days and day < self.days[-1]):
            raise ValueError('Observation date must be a nondecreasing calendar day')
        return ObservationCalendar((*self.days, day))

    def locate(self, day):
        if not self.days or not self.days[0] <= day <= self.days[-1]:
            return None
        return bisect_right(self.days, day) - 1


@dataclass(frozen=True)
class CalendarFocus:
    key: int
    as_of: int
    known_by: int | None
    backfill_event: int | None = None

    @classmethod
    def from_read(cls, route, selection, previous=None):
        if route['reason'] != 'routed':
            return None
        key, as_of = route['key'], route['as_of']
        backfill = None
        if previous is not None and (previous.key, previous.as_of) == (key, as_of):
            backfill = previous.backfill_event
        if (selection is not None and selection['status'] == 'experimental_candidate'
                and selection['effective_time'] < selection['event_time']):
            backfill = selection['event_time']
        return cls(key, as_of, route['known_by'], backfill)


def resolve_calendar_followup(question, calendar, focus):
    """Resolve just one changed time coordinate against the current conversation focus."""
    if not isinstance(question, str) or len(question) > 300:
        return None
    text = _FOLLOW_TAIL.sub('', _FOLLOW_PREFIX.sub('', unicodedata.normalize('NFKC', question.strip()))).strip()
    known = _KNOWLEDGE.fullmatch(text)
    effective = _DATE_ONLY.fullmatch(text)
    present = _PRESENT.fullmatch(text)
    previous = _BEFORE_BACKFILL.fullmatch(text)
    if not (known or effective or present or previous):
        return None
    if focus is None:
        return {'reason': 'date_followup_requires_subject'}
    as_of, known_by = focus.as_of, focus.known_by
    if known or effective:
        try:
            day = _date((known or effective)['date'])
        except ValueError:
            return {'reason': 'invalid_calendar_date'}
        position = calendar.locate(day)
        if position is None:
            return {'reason': 'date_outside_saved_calendar'}
        if known:
            known_by = position
        else:
            as_of = position
    elif previous:
        if focus.backfill_event is None:
            return {'reason': 'date_followup_requires_backfill'}
        known_by = focus.backfill_event - 1
    else:
        known_by = None
    return {'reason': 'routed', 'key': focus.key, 'as_of': as_of, 'known_by': known_by}


def resolve_calendar_question(question, fingerprints, calendar):
    """Return exact routing coordinates for a single enrolled name and supported date request."""
    if not isinstance(question, str) or not question.strip() or len(question) > 300:
        return {'reason': 'unsupported_date_question'}
    text = _ENDING.sub('', unicodedata.normalize('NFKC', question.strip()))
    clauses = _SEPARATOR.split(text)
    if len(clauses) > 2:
        return {'reason': 'unsupported_date_question'}
    known = None
    if len(clauses) == 2:
        for index in (0, 1):
            known_clause = clauses[index].rstrip('。?')
            known = _KNOWLEDGE.fullmatch(known_clause)
            if known or _PRESENT.fullmatch(known_clause):
                clauses = [clauses[1 - index]]
                break
        else:
            return {'reason': 'unsupported_date_question'}
    match = next((match for pattern in _CORE if (match := pattern.fullmatch(clauses[0]))), None)
    if match is None:
        return {'reason': 'unsupported_date_question'}
    try:
        effective_day = _date(match['date'])
        known_day = _date(known['date']) if known else None
    except ValueError:
        return {'reason': 'invalid_calendar_date'}
    name = match['name']
    key = fingerprints.lookup(name)
    if key is None:
        return {'reason': 'date_subject_not_enrolled'}
    as_of = calendar.locate(effective_day)
    known_by = calendar.locate(known_day) if known_day is not None else None
    if as_of is None or (known_day is not None and known_by is None):
        return {'reason': 'date_outside_saved_calendar'}
    return {'reason': 'routed', 'key': key, 'as_of': as_of, 'known_by': known_by}
