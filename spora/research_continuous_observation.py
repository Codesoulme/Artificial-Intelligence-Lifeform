"""Parse bounded, explicitly submitted statements for the research chat."""

import re


ADDRESSES = {'齐宁': 26, '何程': 47}
PREDICATES = {
    'drink': r'(?:最喜欢喝|最爱喝|偏爱喝|(?:最喜欢的|偏爱的)?(?:饮品|饮料)(?:是|改为|改成))',
    'work': r'(?:(?:职业|工作|岗位)(?:是|改为|改成)|从事的工作是|任职为)',
    'place': r'(?:(?:去了|到访了)|(?:到访地点|目的地)(?:是|改为|改成))',
}
CLAUSE = re.compile(
    r'(?P<subject>' + '|'.join(map(re.escape, ADDRESSES)) + r')(?:的)?(?:现在|目前|那次)?'
    + '(?:' + '|'.join(fr'(?P<{field}>{pattern})' for field, pattern in PREDICATES.items()) + ')'
    + r'\s*(?P<value>[\u4e00-\u9fff]{1,18})[。.]?'
)
AMBIGUOUS = re.compile(
    r'齐宁|何程|未知|不详|不确定|不清楚|不知道|可能|也许|大概|或许|疑似|应该|据说|听说|'
    r'并非|不是|没有|不再|不喜欢|没去|是否|是不是|还是|或者|以及|同时|而且|但|如果|假设|才怪|'
    r'也喜欢|最喜欢|最爱喝|偏爱喝|改为|改成|职业是|工作是|岗位是|目的地是|到访地点是|去了|到访了|[吗呢吧么]$'
)


def parse_observation(text):
    """Return one explicit entity, field, and value, or defer the input."""
    match = CLAUSE.fullmatch(text.strip())
    if match is None or AMBIGUOUS.search(match['value']):
        return None
    field = next(field for field in PREDICATES if match[field] is not None)
    return ADDRESSES[match['subject']], field, match['value']
