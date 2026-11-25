"""Build explicit address candidates and scope each version decision to one mention."""

from dataclasses import dataclass
from enum import IntEnum
import re

from torch import nn

from .research_version_store import MemoryAddress
from .research_version_timeline import VersionQuery, parse_version_query, select_version


class Lookup(IntEnum):
    SKIP = 0
    CURRENT = 1
    FIRST = 2
    PREVIOUS = 3


_REQUESTS = {
    Lookup.CURRENT: VersionQuery('current'),
    Lookup.FIRST: VersionQuery('ordinal', 1),
    Lookup.PREVIOUS: VersionQuery('previous'),
}
_BOUNDARY = re.compile(r'[，,。；;！？?!、\n]')
_NAME_LEAD = re.compile(r'(?:^|[「“（(\s，,。；;：:！？?!、]|核对|检查|看看|看一眼|查看|查查|查|查一下|查到|查询'
                        r'|说说|回忆|告诉我|提到|想起了|搁置|关于|以及|并查|再把|给|报|和|与)$')
_READ = re.compile(r'核对|核实|查|回忆|告诉我|说说|请报|是什么|哪[个份本条项种]|第[一二三四五六七八九1-9]版')
_TEXT_TASK = re.compile(r'[「」“”]|翻译|润色|改写|措辞|点评|只评价|都不用|都不必|都别查|先别查档')
_LOCAL_SKIP = re.compile(r'(?:别|不必|不用|不要|不需要)(?:再)?(?:查|读|看|翻)|(?:先|暂)搁(?:下|置|在一边)?|先放一边')
_PREVIOUS_ALIAS = re.compile(r'(?:现在|当前|现行|现用)(?:的)?前一(?:份|版)|上(?:一?个版本|一份记录)|前一份')
_FIRST_ALIAS = re.compile(r'最初|初次登记|当初')
_UNRESOLVED_RELATIVE = re.compile(r'此前|以前|之前|之后|前[一二三几]\S|生效前|更正前')


@dataclass(frozen=True)
class Mention:
    field: str
    start: int
    end: int
    owner: str | None

    def views(self, question):
        marked = question[:self.start] + '【' + question[self.start:self.end] + '】' + question[self.end:]
        before = _BOUNDARY.split(question[:self.start])[-1]
        after = _BOUNDARY.split(question[self.end:])[0]
        local = before + '【' + question[self.start:self.end] + '】' + after
        return marked, local


def mentions(question, enrolled, fields):
    """Bind literal fields to complete enrolled names; unresolved possessives reset scope."""
    pattern = re.compile('|'.join(map(re.escape, sorted(fields.values(), key=len, reverse=True))))
    labels = {label: field for field, label in fields.items()}
    names = sorted(enrolled, key=len, reverse=True)
    owner = None
    result = []
    for found in pattern.finditer(question):
        prefix = question[:found.start()]
        if prefix.endswith('的'):
            owner = next((name for name in names if prefix.endswith(name + '的') and
                          _NAME_LEAD.search(prefix[:-len(name) - 1])), None)
        elif result and re.search(r'[。！？?!\n]', question[result[-1].end:found.start()]):
            owner = None
        result.append(Mention(labels[found[0]], found.start(), found.end(), owner))
    return result


class LookupHead(nn.Sequential):
    def __init__(self, width):
        super().__init__(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, len(Lookup)))


def resolve(catalog, mention, request, today):
    if request == Lookup.SKIP:
        return None, 'not_requested'
    return resolve_query(catalog, mention, _REQUESTS[request], today)


def resolve_query(catalog, mention, query, today):
    if query is None:
        return None, 'unresolved_scope'
    address = MemoryAddress(mention.owner, mention.field) if mention.owner is not None else None
    versions = catalog.get(address, ())
    if not versions:
        return None, 'unknown_address'
    timeline = tuple((entry.effective, entry.observed) for entry in versions)
    selected = select_version(timeline, query, today)
    return (None, 'missing_version') if selected is None else (versions[selected].number, 'located')


def explicit_queries(question, candidates):
    """Use explicit timeline constraints, holding ambiguous request clauses for clarification."""
    if _TEXT_TASK.search(question) or not _READ.search(question):
        return [Lookup.SKIP for _ in candidates]
    queries = []
    for item in candidates:
        scope = item.views(question)[1]
        if _LOCAL_SKIP.search(scope):
            queries.append(Lookup.SKIP)
            continue
        normalized = _FIRST_ALIAS.sub('首版', _PREVIOUS_ALIAS.sub('上一版', scope))
        query = parse_version_query(normalized)
        if query is not None and query.kind == 'current' and _UNRESOLVED_RELATIVE.search(normalized):
            query = None
        queries.append(query)
    return queries
