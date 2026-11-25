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
_TEXT_TASK = re.compile(r'[「」“”]|翻译|润色|改写|措辞|点评|只评价|都不用|都不必|都别查|先别查档|'
                        r'(?:把|将)(?:这|那)(?:句(?:话)?|段(?:话|文字)?)(?:改得|改成|说得|写得|换个说法)')
_LOCAL_SKIP = re.compile(r'(?:别|不必|不用|不要|不需要)(?:再)?(?:查|读|看|翻)|(?:先|暂)搁(?:下|置|在一边)?|先放一边')
_PREVIOUS_ALIAS = re.compile(r'(?:现在|当前|现行|现用)(?:的)?前一(?:份|版)|上(?:一?个版本|一份记录)|前一份')
_FIRST_ALIAS = re.compile(r'最初|初次登记|当初')
_UNRESOLVED_RELATIVE = re.compile(r'此前|以前|之前|之后|前[一二三几]\S|生效前|更正前')
_UNCERTAIN_LOOKUP = re.compile(r'还是|或者|抑或|任选|随便|不知道|不确定|记不清|都可以|任意|说不清|不作数|不要|不是|\d|'
                               r'[零一二三四五六七八九十廿]+月|第[两二三四五六七八九十]|截至|到期|哪个版本')
_FIRST_MARKER = re.compile(r'第一笔|第一本登记|头一[回遭]|一上来|最开[始头]|起头|开头')
_ENTRY_MARKER = re.compile(r'登|录|入档|立档|存档|清单|档案|写进|记进|头一')
_ADJACENT = re.compile(r'紧挨|紧贴|挨着|前一格|往前挪一档|往前推一[格个档]|向前数一条|接着的上一条')
_LATEST_CHANGE = re.compile(r'最近一次|最后一次|刚(?:换|改|调|替|被)')
_CHANGED_OUT = re.compile(r'替掉|换下|换走|调换下|之前|以前|改动前|生效前')
_ACTIVE = re.compile(r'仍有效|眼下有效|还在照办|还管用|仍在用|手头还认|还算有效|现用|正在用')


@dataclass(frozen=True)
class Mention:
    field: str
    start: int
    end: int
    owner: str | None
    possessive: bool = False

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
        result.append(Mention(labels[found[0]], found.start(), found.end(), owner, prefix.endswith('的')))
    return result


class LookupHead(nn.Sequential):
    def __init__(self, width):
        super().__init__(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Linear(64, len(Lookup)))


def resolve(catalog, mention, request, today):
    if request == Lookup.SKIP:
        return None, 'not_requested'
    return resolve_query(catalog, mention, _REQUESTS[request], today)


def resolve_query(catalog, mention, query, today):
    address = MemoryAddress(mention.owner, mention.field) if mention.owner is not None else None
    return resolve_address(catalog, address, query, today)


def resolve_address(catalog, address, query, today, *, retain_same_day=False):
    if query is None:
        return None, 'unresolved_scope'
    versions = catalog.get(address, ())
    if not versions:
        return None, 'unknown_address'
    timeline = tuple((entry.effective, entry.observed) for entry in versions)
    selected = select_version(timeline, query, today, retain_same_day=retain_same_day)
    return (None, 'missing_version') if selected is None else (versions[selected].number, 'located')


def parse_lookup_scope(scope):
    normalized = _FIRST_ALIAS.sub('首版', _PREVIOUS_ALIAS.sub('上一版', scope))
    query = parse_version_query(normalized)
    if query is not None and query.kind == 'current' and _UNRESOLVED_RELATIVE.search(normalized):
        return None
    return query


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
        queries.append(parse_lookup_scope(scope))
    return queries


def fallback_evidence(question, mention):
    """Isolate wording for an unresolved exact address, without exposing a stored value."""
    before = _BOUNDARY.split(question[:mention.start])[-1]
    suffix = _BOUNDARY.split(question[mention.end:])[0]
    if mention.owner is not None and before.endswith(mention.owner + '的'):
        before = before[:-len(mention.owner) - 1]
    return spoken_scope_evidence(suffix, prefix=before)


def spoken_scope_evidence(scope, *, prefix=''):
    if _UNCERTAIN_LOOKUP.search(prefix + scope) or not scope.strip():
        return None
    return prefix + '该项' + scope


def normalize_version(request):
    """Translate a confident learned proposal into a timeline operation."""
    return _REQUESTS.get(request)


def corroborated_version(request, evidence):
    """Require wording that distinguishes the proposed version from the other versions."""
    support = {
        Lookup.FIRST: _FIRST_MARKER.search(evidence) and _ENTRY_MARKER.search(evidence),
        Lookup.PREVIOUS: _ADJACENT.search(evidence) or (_LATEST_CHANGE.search(evidence) and _CHANGED_OUT.search(evidence)),
        Lookup.CURRENT: _ACTIVE.search(evidence),
    }
    return _REQUESTS.get(request) if support.get(request) else None
