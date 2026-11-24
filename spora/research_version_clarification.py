"""Keep a clarification in working state until the next user turn."""

from dataclasses import dataclass
from datetime import date
from enum import IntEnum
import re
import unicodedata

from torch import nn

from .research_version_roles import apply_version_plan
from .research_version_scope import _DATE_HINT, _ORDINAL_HINT
from .research_version_timeline import VersionQuery, select_version


class Intent(IntEnum):
    READ = 0
    ASK = 1
    DEFER = 2
    OTHER = 3


class VersionAccessHead(nn.Sequential):
    """Estimate whether the utterance requests any versioned-memory action."""

    def __init__(self, width):
        super().__init__(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))


def scope_version_intent(intent, *, permits_lookup):
    """A rejected request can neither read a record nor create a pending lookup."""
    if not permits_lookup and intent in (Intent.READ, Intent.ASK):
        return Intent.OTHER
    return intent


_VERSIONS = (VersionQuery('current'), VersionQuery('ordinal', 1), VersionQuery('previous'))
_LABELS = ('现用版', '最初登记版', '现用前一版')
_WORDS = ('现用版|现在的|当前版|现行版|现在这份', '首版|初版|第一版|最初登记版|最早的',
          '上一版|前一版|现用前一版|上一份|刚换掉的那份')
_REPLIES = tuple(re.compile(r'(?:就(?:查|用|看)|查|用|看|选|要)?(?:' + words + r')(?:吧|就行|即可)?')
                 for words in _WORDS)
_CANCEL = re.compile(r'算了|还是算了|先不查了?|暂时不查|先不看了?|改天再查|下次再说|不用了')
_EXPLICIT_READ = re.compile(r'(?:请)?(?:查|查看|查阅|核对|读取)(?:当前版|现行版|首版|初版|第一版|上一版|前一版)(?:吧|即可)?')


@dataclass(frozen=True)
class PendingVersion:
    subject: int
    revisions: tuple[tuple[date, date], ...]
    today: date
    offered: tuple[int, ...]

    @property
    def question(self):
        return '你想查' + '、'.join(_LABELS[i] for i in self.offered[:-1]) + '，还是' + _LABELS[self.offered[-1]] + '？'


@dataclass(frozen=True)
class VersionDecision:
    action: Intent
    route: tuple[int, int] | None = None
    pending: PendingVersion | None = None


def start_version_turn(question, subject, revisions, today, legacy, plan, intent):
    if intent in (Intent.DEFER, Intent.OTHER):
        return VersionDecision(intent)
    if subject is None:
        return VersionDecision(Intent.OTHER)
    selected, reason = apply_version_plan(question, revisions, today, legacy, plan)
    if reason == 'outside_lookup_scope':
        return VersionDecision(Intent.OTHER)
    if intent == Intent.READ:
        return VersionDecision(Intent.OTHER) if selected is None else VersionDecision(Intent.READ, (subject, selected))
    text = unicodedata.normalize('NFKC', question)
    if _DATE_HINT.search(text) or _ORDINAL_HINT.search(text):
        return VersionDecision(Intent.OTHER)
    # Several temporal descriptions can refer to the same effective version.
    distinct = {}
    for index, query in enumerate(_VERSIONS):
        slot = select_version(revisions, query, today)
        if slot is not None:
            distinct.setdefault(slot, index)
    if len(distinct) < 2:
        return VersionDecision(Intent.OTHER)
    pending = PendingVersion(subject, tuple(revisions), today, tuple(distinct.values()))
    return VersionDecision(Intent.ASK, pending=pending)


def use_explicit_read_plan(question, plan):
    """Keep the timeline interpretation of a standalone, explicit lookup instruction."""
    clauses = re.split(r'[。；.!?;]', unicodedata.normalize('NFKC', question))
    return None if any(_EXPLICIT_READ.fullmatch(part.strip()) for part in clauses) else plan


def resolve_version_turn(pending, reply, timelines, today):
    """Consume the pending turn, accepting only an offered, unambiguous reply."""
    if pending is None:
        return VersionDecision(Intent.OTHER)
    text = unicodedata.normalize('NFKC', reply).strip().rstrip('.!?。').strip()
    if _CANCEL.fullmatch(text):
        return VersionDecision(Intent.DEFER)
    revisions = timelines.get(pending.subject)
    if revisions is None or tuple(revisions) != pending.revisions or today != pending.today:
        return VersionDecision(Intent.OTHER)
    for index in pending.offered:
        if _REPLIES[index].fullmatch(text):
            slot = select_version(revisions, _VERSIONS[index], today)
            return VersionDecision(Intent.READ, (pending.subject, slot))
    return VersionDecision(Intent.OTHER)
