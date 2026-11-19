"""Parse bounded, explicitly submitted statements for the research chat."""

import re
from functools import lru_cache


ADDRESSES = {'齐宁': 26, '何程': 47}
REGISTERED_NAME = re.compile(r'[\u4e00-\u9fff]{2,4}')
PREDICATES = {
    'drink': r'(?:最喜欢喝|最爱喝|偏爱喝|(?:最喜欢的|偏爱的)?(?:饮品|饮料)(?:是|改为|改成))',
    'work': r'(?:(?:职业|工作|岗位)(?:是|改为|改成)|从事的工作是|任职为)',
    'place': r'(?:(?:去了|到访了)|(?:到访地点|目的地)(?:是|改为|改成))',
}
AMBIGUOUS = re.compile(
    r'未知|不详|不确定|不清楚|不知道|可能|也许|大概|或许|疑似|应该|据说|听说|'
    r'并非|不是|没有|不再|不喜欢|没去|是否|是不是|还是|或者|以及|同时|而且|但|如果|假设|才怪|'
    r'也喜欢|最喜欢|最爱喝|偏爱喝|改为|改成|职业是|工作是|岗位是|目的地是|到访地点是|去了|到访了|[吗呢吧么]$'
)
ELLIPTICAL_QUERY = re.compile(
    r'(?:那|那么)?(?:查到了就说一下|说来听听|这项具体写了什么|具体内容呢|那一项的答案是什么|'
    r'你能说出那款的名字吗|能再告诉我这个细节吗|请把这个细节说给我|具体是什么|那项是什么)[。？！.!?]?'
)
HISTORY_PREFIX = re.compile(
    r'^(?:请)?(?:查(?:一下)?|回看)?(?:上一次记录(?:里|中)?|'
    r'(?P<count>[0-9零一二两三四五六七八九十]+)次更正前)(?=[，,：:\s]|$)'
)
CHINESE_DIGITS = '零一二三四五六七八九'


def split_history_query(text):
    """Remove an explicit history prefix and return its correction offset."""
    match = HISTORY_PREFIX.match(text.strip())
    if match is None:
        return text, 0
    question = text.strip()[match.end():].lstrip('，,：: \t\n\r')
    count = (match['count'] or '一').replace('两', '二')
    if count.isascii() and len(count) <= 2:
        offset = int(count)
    elif len(count) == 1 and count in CHINESE_DIGITS:
        offset = CHINESE_DIGITS.index(count)
    elif re.fullmatch(r'[一二三四五六七八九]?十[一二三四五六七八九]?', count):
        tens, units = count.split('十')
        offset = 10 * CHINESE_DIGITS.index(tens or '一') + CHINESE_DIGITS.index(units or '零')
    else:
        offset = 0
    return (question, offset) if question and offset else (text, None)


@lru_cache(maxsize=16)
def subject_patterns(names):
    alternatives = '|'.join(map(re.escape, names))
    subject = r'(?P<subject>' + alternatives + ')'
    fields = '|'.join(fr'(?P<{field}>{pattern})' for field, pattern in PREDICATES.items())
    clause = re.compile(subject + r'(?:的)?(?:现在|目前|那次)?(?:' + fields + r')\s*(?P<value>[\u4e00-\u9fff]{1,18})[。.]?')
    query = re.compile(
        r'(?:^|[^\u4e00-\u9fff]|关于|想查|查一下|确认|想问|知道|提到|翻了|说说|看看|问问|'
        r'改问|问的是|告诉我|线索是|谈的是|说的是|记录里)' + subject
        + r'(?=$|[^\u4e00-\u9fff]|的|现在|目前|平常|当时|以前|平时|曾经|那次|那回|那天|在座谈|'
        r'最|偏爱|爱喝|喜欢|做|去了|到访|担任|从事)'
    )
    return clause, query


def can_register(name, addresses):
    return bool(REGISTERED_NAME.fullmatch(name)) and not any(name in other or other in name for other in addresses)


def parse_observation(text, addresses=ADDRESSES):
    """Return one explicit entity, field, and value, or defer the input."""
    clause, _ = subject_patterns(tuple(addresses))
    match = clause.fullmatch(text.strip())
    if match is None or AMBIGUOUS.search(match['value']) or any(name in match['value'] for name in addresses):
        return None
    field = next(field for field in PREDICATES if match[field] is not None)
    return addresses[match['subject']], field, match['value']


def query_subject(text, addresses=ADDRESSES):
    """Resolve a bounded exact mention. Return an address and the match status."""
    text = text.strip()
    mentions = [name for name in addresses if name in text]
    if len(mentions) > 1:
        return None, 'ambiguous_subject'
    if not mentions:
        return None, 'no_subject'
    _, query = subject_patterns(tuple(addresses))
    match = query.search(text)
    if match is None:
        return None, 'unresolved_subject'
    return addresses[match['subject']], 'named_subject'


def inherited_query_subject(question, history, addresses=ADDRESSES):
    if history and ELLIPTICAL_QUERY.fullmatch(question.strip()):
        entity, _ = query_subject(history, addresses)
        return entity
    return None
