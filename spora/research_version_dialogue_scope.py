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
_CANCEL_TURN = re.compile(r'^(?:先别查(?:了)?|先不查(?:了)?|不查了|算了|先暂停)(?:[，,。；;！!\s]|$)')
_REJECTED_CLAUSE = re.compile(r'^(?:不是|别查|不查|不用查|不要查|先不看)')
_AMBIGUOUS_REFERENCE = re.compile(r'谁的|(?:他|她|对方|另一个人|别人)的')
_ANCHORED_PERSON = re.compile(r'\S+的(?=(?:首|初|上一|前一|当前|现行|第[一二三四五六七八九1-9])版)')
_PENDING_CHOICE = re.compile(r'(?:就|请)?(?:查|看|用|选)?(?:当前版|现行版|首版|初版|第一版|上一版|前一版)(?:吧|即可)?[。！？?!]?')
_LOOKUP_START = re.compile(r'^(?:请|帮我|麻烦)?(?:查一下|查询|核对|查)')
_VERSION_WORD = r'当前版|现行版|首版|初版|上一版|前一版|第[一二三四五六七八九1-9]版'
_SLOT_REPLY = re.compile(r'^(?:(?:就|请|先)(?:查|看|选)|说的是|改查|改成|用|是|查)?\s*')
_CORRECTION = re.compile(r'^不是(.+?)[，,]\s*(?:而?是|改查|改成)(.+)$')
_ALTERNATIVE = re.compile(r'还是|或者|抑或|或')
_TRAILING = '。！？?!吧'


@dataclass(frozen=True)
class LookupTarget:
    address: MemoryAddress | None
    query: VersionQuery | None
    source: str
    evidence: str | None = None


@dataclass(frozen=True)
class PartialLookup:
    owner: str | None = None
    field: str | None = None
    query: VersionQuery | None = None

    def target(self):
        if self.owner is None or self.field is None or self.query is None:
            return None
        return LookupTarget(MemoryAddress(self.owner, self.field), self.query, 'confirmed')


def _lookup_parts(text, enrolled, fields):
    """Parse a short whole utterance with at most one owner, field and version."""
    labels = {label: field for field, label in fields.items()}
    names = '|'.join(map(re.escape, sorted(enrolled, key=len, reverse=True)))
    field_labels = '|'.join(map(re.escape, sorted(labels, key=len, reverse=True)))
    match = re.fullmatch(fr'(?:(?P<name>谁|{names or "(?!)"})(?:的)?\s*)?'
                         fr'(?:(?P<field>{field_labels})\s*)?'
                         fr'(?P<version>{_VERSION_WORD})?(?P<record>记录|档案)?',
                         text.strip().rstrip(_TRAILING).strip())
    if match is None or not any(match.groups()):
        return None
    name, field, version, _ = match.groups()
    return PartialLookup(name if name != '谁' else None, labels.get(field), parse_lookup_scope(version) if version else None)


def start_partial_lookup(question, enrolled, fields):
    """Remember missing coordinates only when the user explicitly asks to look up a record."""
    opening = _LOOKUP_START.match(question.strip())
    if opening is None:
        return None
    request = _lookup_parts(question.strip()[opening.end():], enrolled, fields)
    return request if request is not None and (request.owner is None or request.field is None) else None


def ambiguous_partial_reply(question, enrolled, fields):
    parts = _ALTERNATIVE.split(_SLOT_REPLY.sub('', question.strip(), count=1))
    if len(parts) < 2:
        return None
    replies = [_lookup_parts(text, enrolled, fields) for text in parts]
    if any(reply is None for reply in replies):
        return None
    for slot in ('owner', 'field', 'query'):
        choices = {getattr(reply, slot) for reply in replies}
        if None not in choices and len(choices) > 1:
            return slot
    return None


def continue_partial_lookup(question, pending, enrolled, fields):
    text = question.strip().rstrip(_TRAILING).strip()
    correction = _CORRECTION.fullmatch(text)
    if correction:
        old, text = correction.groups()
        rejected = _lookup_parts(old, enrolled, fields)
        replacement = _lookup_parts(text, enrolled, fields)
        if (rejected is None or replacement is None or rejected == replacement
                or not any(previous is not None and previous == getattr(pending, slot)
                           and getattr(replacement, slot) is not None
                           for slot, previous in zip(('owner', 'field', 'query'),
                                                     (rejected.owner, rejected.field, rejected.query), strict=True))):
            return None
        reply = replacement
    else:
        reply = _lookup_parts(_SLOT_REPLY.sub('', text, count=1), enrolled, fields)
    if reply is None or reply == PartialLookup():
        return None
    changed_address = reply.owner and reply.field and (
        (pending.owner and reply.owner != pending.owner) or (pending.field and reply.field != pending.field)
    )
    version = reply.query if changed_address and not correction else reply.query or pending.query
    return PartialLookup(reply.owner or pending.owner, reply.field or pending.field, version)


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
    if _PENDING_CHOICE.fullmatch(question.strip()):
        return (LookupTarget(address, parse_lookup_scope(question), 'pending'),)
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
