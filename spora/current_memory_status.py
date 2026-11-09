"""Read-only known/unknown status for a finite SPORA current-memory slot."""
import torch

from .growing import query


@torch.no_grad()
def current_memory_status(model, state, key):
    """Return native known/unknown prediction without returning the stored value."""
    if type(key) is not int or not 0 <= key < model.config.key_count:
        raise ValueError('Key outside finite memory domain')
    device = next(model.parameters()).device
    keys = torch.tensor([key], dtype=torch.long, device=device)
    zeros = torch.zeros_like(keys)
    logits = query(model, state, keys, zeros, zeros)
    if logits.shape != (1, model.config.value_count) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Invalid native memory status logits')
    probabilities = logits.float().softmax(-1)[0]
    return {'status': 'unknown' if int(probabilities.argmax()) == model.config.unknown else 'known',
            'unknown_probability': float(probabilities[model.config.unknown])}
