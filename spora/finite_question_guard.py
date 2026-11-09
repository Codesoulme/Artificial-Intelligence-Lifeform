"""Conservative format validation for experimental finite numbered-memory queries.

This does not infer intent, retrieve memory, or support arbitrary entity names.
"""
import re

_ENTITY = re.compile(r'(?:对象|编号)\s*([0-9]+)')


def valid_finite_question_form(question, key_count=64):
    """Accept one canonical in-range Arabic identifier and no other digit tokens."""
    if not isinstance(question, str) or type(key_count) is not int or key_count <= 0:
        raise ValueError('Expected a question string and positive integer key count')
    matches = list(_ENTITY.finditer(question))
    if len(matches) != 1:
        return False
    match = matches[0]
    digits = match.group(1)
    if (len(digits) > 1 and digits.startswith('0')) or len(digits) > len(str(key_count - 1)):
        return False
    if int(digits) >= key_count:
        return False
    remainder = question[:match.start(1)] + question[match.end(1):]
    return not any(char.isdigit() for char in remainder)
