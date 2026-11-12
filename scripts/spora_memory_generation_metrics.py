"""Strict development scoring for the authored fruit-memory probes."""


def asserted_fact(answer, name, expected):
    return answer.strip() in (expected, f'{name}喜欢的水果是{expected}。')


def score_pair(own, swapped, *, name, original, replacement):
    if original == replacement:
        raise ValueError('Counterfactual facts must differ')
    own_correct = asserted_fact(own, name, original)
    swap_correct = asserted_fact(swapped, name, replacement)
    return {'own_correct': own_correct, 'swapped_correct': swap_correct,
            'pair_correct': own_correct and swap_correct,
            'original_still_present': original in swapped}
