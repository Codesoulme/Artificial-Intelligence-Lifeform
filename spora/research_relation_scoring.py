"""Strict, isolated scoring of controlled Chinese relation-answer templates.

This is exact template matching, not open-domain semantic understanding. Only a
single complete sentence is accepted; extra clauses, explanations, alternative
values and unsupported paraphrases are rejected. Subjects and values must be
simple, unambiguous tokens. A rejection is not a diagnosis of why an answer is
wrong. No model, user data, or production scoring code is accessed.

Supported relation keys: 饮品, 颜色, 活动. For example::

    score_relation_answer("小林喜欢喝酸奶。", "小林", "饮品", "酸奶")
"""

import re


# Every alternative includes the subject, the relationship, and the whole value.
_TEMPLATES = {
    "饮品": ("{subject}喜欢喝{value}", "{subject}喜欢喝的是{value}", "{subject}喜欢喝的饮料是{value}", "{subject}喜欢的饮品是{value}",
             "{subject}喜欢的饮料是{value}"),
    "颜色": ("{subject}喜欢的颜色是{value}",),
    "活动": ("{subject}喜欢的活动是{value}",),
}
_TOKEN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9·]{1,30}\Z")
# These words make it unsafe to treat caller-supplied strings as atomic names or
# values. Restricting them sacrifices recall, which is intentional here.
_UNSAFE = ("不", "没", "无", "或者", "还是", "以及", "可能", "也许", "和", "与", "或")


def _valid_token(value: object) -> bool:
    return (isinstance(value, str) and bool(_TOKEN.fullmatch(value))
            and not any(word in value for word in _UNSAFE))


def score_relation_answer(answer: str, subject: str, relation: str, value: str) -> bool:
    """Return whether *answer* exactly expresses the requested controlled fact.

    Whitespace and one final Chinese or ASCII sentence terminator are allowed.
    Unsupported relation keys or unsafe/empty subject and value inputs
    return False. In particular, bare values, multiple sentences and negation
    cannot pass merely because the expected text appears within them.

    The short form “X喜欢Y” is inherently relation-ambiguous and is therefore
    deliberately not accepted for color or activity; use the explicit forms.
    """
    if not isinstance(answer, str) or not _valid_token(subject) or not _valid_token(value):
        return False
    if not isinstance(relation, str) or relation not in _TEMPLATES:
        return False
    answer = re.sub(r"\s+", "", answer)
    if answer.endswith(("。", ".", "！", "!")):
        answer = answer[:-1]
    templates = _TEMPLATES[relation]
    return any(answer == template.format(subject=subject, value=value)
               for template in templates)
