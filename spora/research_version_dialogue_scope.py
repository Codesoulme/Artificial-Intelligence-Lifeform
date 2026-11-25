"""Bind a follow-up lookup to the addresses confirmed in the active dialogue."""

from dataclasses import dataclass, replace
import re

from .research_multi_version_route import (Lookup, _LOCAL_SKIP, _TEXT_TASK, explicit_queries, fallback_evidence, mentions,
                                           parse_lookup_scope, spoken_scope_evidence)
from .research_version_store import MemoryAddress
from .research_version_timeline import VersionQuery


_FOLLOW = re.compile(r'^(?:那(?:么|就)?|再|接着|也|这次|同样|两项都|两条都|两个都|都|就查)')
_VERSION_HINT = re.compile(r'版|档|值|记录|用过|登记|换[下掉]|替[下掉]|写的|起头|头一|仍有效|现在|当时')
_PAIR = re.compile(r'两项|两条|两个')
_RESET = re.compile(r'换(?:个|一个)?(?:话题|问题)|转到|转题|聊点别的|先别查|先不查|不查了|算了|先放下|先暂停')
_CANCEL_TURN = re.compile(r'^(?:先别查|先不查(?:了)?|不查了|算了|先暂停)(?:[，,。；;！!\s]|$)')
_REJECTED_CLAUSE = re.compile(r'^(?:不是|别查|不查|不用查|不要查|先不看)')
_AMBIGUOUS_REFERENCE = re.compile(r'谁的|(?:他|她|对方|另一个人|别人)的')
_ANCHORED_PERSON = re.compile(r'\S+的(?=(?:首|初|上一|前一|当前|现行|第[一二三四五六七八九1-9])版)')


@dataclass(frozen=True)
class LookupTarget:
    address: MemoryAddress | None
    query: VersionQuery | None
    source: str
    evidence: str | None = None


def owner_for_field(scope, field):
    matching = tuple(address for address in scope if address.field == field)
    if len(matching) == 1:
        return matching[0]
    owners = {address.name for address in scope}
    if not matching and len(owners) == 1 and all(not address.condition for address in scope):
        return MemoryAddress(next(iter(owners)), field)
    return None


def dialogue_targets(question, enrolled, fields, scope, *, bounded=True):
    """Form a proposed read. The caller still validates every address and version."""
    text = question.strip()
    if _TEXT_TASK.search(text) or _CANCEL_TURN.match(text):
        return ()
    follow = bool(_FOLLOW.match(text) and _VERSION_HINT.search(text))
    resetting = bool(_RESET.search(text))
    if bounded and resetting and not follow:
        scope = ()
    candidates = mentions(text, enrolled, fields)
    if not candidates:
        if not follow or (resetting and bounded):
            return ()
        if (_AMBIGUOUS_REFERENCE.search(text) or _ANCHORED_PERSON.search(text) or not scope
                or (len(scope) > 1 and not _PAIR.search(text))
                or (len(scope) == 1 and _PAIR.search(text))):
            return (LookupTarget(None, None, 'unresolved_reference'),)
        query = parse_lookup_scope(text)
        evidence = spoken_scope_evidence(text) if query is None else None
        return tuple(LookupTarget(address, query, 'carried', evidence) for address in scope)

    queries = explicit_queries(text, candidates)
    targets = []
    for item, query in zip(candidates, queries, strict=True):
        clause = item.views(text)[1].strip()
        if _LOCAL_SKIP.search(clause) or _REJECTED_CLAUSE.match(clause):
            continue
        if query == Lookup.SKIP:
            if not follow or (bounded and resetting):
                continue
            query = parse_lookup_scope(clause)
        if item.owner is not None:
            address, source = MemoryAddress(item.owner, item.field), 'named'
        elif not item.possessive and follow and not resetting and not _AMBIGUOUS_REFERENCE.search(clause):
            address, source = owner_for_field(scope, item.field), 'carried'
        else:
            address, source = None, 'unresolved_reference'
        evidence = fallback_evidence(text, item) if query is None else None
        targets.append(LookupTarget(address, query, source if address is not None else 'unresolved_reference', evidence))
    return tuple(targets)


def advance_scope(scope, targets, selected, *, persistent=False):
    """Only actual complete reads can confirm a new address scope."""
    if selected and len(selected) == len(targets) and all(item.address is not None for item in targets):
        return tuple(dict.fromkeys(item.address for item in targets))
    return scope if persistent else ()


def pending_followup_targets(question, enrolled, fields, address):
    targets = dialogue_targets(question, enrolled, fields, (address,))
    result = []
    for item in targets:
        if item.source == 'carried':
            if item.address == address:
                item = replace(item, source='pending')
            else:
                item = replace(item, address=None, source='unresolved_reference')
        result.append(item)
    return tuple(result)


def pending_after_lookup(catalog, targets, selected, reasons):
    """One pending address can survive only a first unresolved version question."""
    if selected or len(targets) != 1 or reasons != ('unresolved_scope',):
        return None
    target = targets[0]
    return target.address if target.source != 'pending' and target.address in catalog else None
