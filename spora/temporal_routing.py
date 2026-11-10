"""Small, deterministic mapping from temporal intent to memory page."""

CURRENT, PAST, LATER = 0, 1, 2


def route_spec(intent):
    if intent == CURRENT:
        return 0, None
    if intent == PAST:
        return 1, 0
    if intent == LATER:
        return 1, 1
    raise ValueError(f'Unknown temporal intent: {intent}')
