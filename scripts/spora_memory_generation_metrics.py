"""Development scoring for authored fruit-memory probes."""


_LIST_MARKERS = ('包括', '例如', '比如', '还有', '、', '和')
_NEGATION_MARKERS = ('不喜欢', '不爱吃', '不是')
_FACT_MARKERS = ('喜欢', '爱吃', '偏好', '水果是', '水果为', '水果的偏好')


def asserted_fact(answer, name, expected):
    text = answer.strip()
    if text == expected:
        return True
    if not text or expected not in text or name not in text:
        return False
    if any(marker in text for marker in _LIST_MARKERS + _NEGATION_MARKERS):
        return False
    return any(marker in text for marker in _FACT_MARKERS)


def score_pair(own, swapped, *, name, original, replacement):
    if original == replacement:
        raise ValueError('Counterfactual facts must differ')
    own_correct = asserted_fact(own, name, original)
    swap_correct = asserted_fact(swapped, name, replacement)
    return {'own_correct': own_correct, 'swapped_correct': swap_correct,
            'pair_correct': own_correct and swap_correct,
            'original_still_present': original in swapped}
