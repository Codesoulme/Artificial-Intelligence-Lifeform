"""Bind incremental attention state to the weights and memory it was built from."""
import torch

from .sleep_memory import SealedPage


def _sources(model, state):
    tensors = list(model.parameters()) + list(state.current)
    for page in state.pages:
        if isinstance(page, SealedPage):
            tensors.extend(t for left, right, _ in page.factors for t in (left, right))
        else:
            tensors.extend(site[0] for site in page)
    if any(torch.is_inference(t) for t in tensors):
        raise ValueError('Incremental decoding requires versioned weights and memory tensors')
    return tensors


class ConversationPast(list):
    """Ephemeral state; restart a full prefix after a memory/weight update.

    Tensor version checks cover normal PyTorch updates and replacements. Direct
    .data mutation or foreign writes bypassing version counters are unsupported.
    This is a consistency check, not persistent autobiographical storage.
    """
    def __init__(self, layers, model, state):
        super().__init__(layers)
        self.owner = model
        self.cursor = state.cursor
        self.sources = tuple((t, t._version) for t in _sources(model, state))

    def validate(self, model, state):
        tensors = _sources(model, state)
        if (self.owner is not model or self.cursor != state.cursor
                or len(tensors) != len(self.sources)
                or any(t is not old or t._version != version
                       for t, (old, version) in zip(tensors, self.sources))):
            raise ValueError('Weights or memory changed; rebuild the complete conversation prefix')
        if len(self) != model.config.depth:
            raise ValueError('Incomplete incremental attention state')
