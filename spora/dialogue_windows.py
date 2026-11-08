"""Overlapping context windows with each assistant target supervised once."""


def dialogue_windows(tokens, mask, length=512, overlap=128):
    if len(tokens) != len(mask) or len(tokens) < 2:
        raise ValueError('Expected aligned dialogue tokens')
    if not isinstance(length, int) or not isinstance(overlap, int) or not 0 < overlap < length:
        raise ValueError('Invalid window dimensions')
    if any(type(x) is not bool for x in mask):
        raise ValueError('Mask must contain booleans')
    previous_end, start = 0, 0
    while start < len(tokens) - 1:
        end = min(start + length, len(tokens))
        local_mask = [bool(mask[i]) and i >= previous_end and i > start for i in range(start, end)]
        if any(local_mask[1:]):
            yield start, tokens[start:end], local_mask
        if end == len(tokens):
            break
        previous_end, start = end, end - overlap
