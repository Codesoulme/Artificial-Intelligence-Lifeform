"""Parse bounded, explicitly submitted statements for the research chat."""

import re


ADDRESSES = {'齐宁': 26, '何程': 47}
NAMES = '|'.join(map(re.escape, ADDRESSES))
PREDICATES = {
    'drink': r'(?:最喜欢喝|最爱喝|偏爱喝|(?:最喜欢的|偏爱的)?(?:饮品|饮料)(?:是|改为|改成))',
    'work': r'(?:(?:职业|工作|岗位)(?:是|改为|改成)|从事的工作是|任职为)',
    'place': r'(?:(?:去了|到访了)|(?:到访地点|目的地)(?:是|改为|改成))',
}
CLAUSE = re.compile(
    r'(?P<subject>' + NAMES + r')(?:的)?(?:现在|目前|那次)?'
    + '(?:' + '|'.join(fr'(?P<{field}>{pattern})' for field, pattern in PREDICATES.items()) + ')'
    + r'\s*(?P<value>[\u4e00-\u9fff]{1,18})[。.]?'
)
AMBIGUOUS = re.compile(
    r'齐宁|何程|未知|不详|不确定|不清楚|不知道|可能|也许|大概|或许|疑似|应该|据说|听说|'
    r'并非|不是|没有|不再|不喜欢|没去|是否|是不是|还是|或者|以及|同时|而且|但|如果|假设|才怪|'
    r'也喜欢|最喜欢|最爱喝|偏爱喝|改为|改成|职业是|工作是|岗位是|目的地是|到访地点是|去了|到访了|[吗呢吧么]$'
)
NAME_MENTION = re.compile(NAMES)
NAME_IN_QUERY = re.compile(
    r'(?:^|[^\u4e00-\u9fff]|关于|想查|查一下|确认|想问|知道|提到|翻了|说说|看看|问问|'
    r'改问|问的是|告诉我|线索是|谈的是|说的是|记录里)'
    r'(?P<subject>' + NAMES + r')(?=$|[^\u4e00-\u9fff]|的|现在|目前|平常|当时|以前|平时|曾经|'
    r'那次|那回|那天|在座谈|最|偏爱|爱喝|喜欢|做|去了|到访|担任|从事)'
)
ELLIPTICAL_QUERY = re.compile(
    r'(?:那|那么)?(?:查到了就说一下|说来听听|这项具体写了什么|具体内容呢|那一项的答案是什么|'
    r'你能说出那款的名字吗|能再告诉我这个细节吗|请把这个细节说给我|具体是什么|那项是什么)[。？！.!?]?'
)


def parse_observation(text):
    """Return one explicit entity, field, and value, or defer the input."""
    match = CLAUSE.fullmatch(text.strip())
    if match is None or AMBIGUOUS.search(match['value']):
        return None
    field = next(field for field in PREDICATES if match[field] is not None)
    return ADDRESSES[match['subject']], field, match['value']


def query_subject(text):
    """Resolve a bounded exact mention. Return an address and the match status."""
    mentions = set(NAME_MENTION.findall(text.strip()))
    if len(mentions) > 1:
        return None, 'ambiguous_subject'
    if not mentions:
        return None, 'no_subject'
    match = NAME_IN_QUERY.search(text.strip())
    if match is None:
        return None, 'unresolved_subject'
    return ADDRESSES[match['subject']], 'named_subject'


def inherited_query_subject(question, history):
    if history and ELLIPTICAL_QUERY.fullmatch(question.strip()):
        entity, _ = query_subject(history)
        return entity
    return None
