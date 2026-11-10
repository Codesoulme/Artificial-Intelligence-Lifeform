"""Combine independent one-row growing states for batched read-only inference."""
from dataclasses import dataclass

import torch

from .sleep_memory import history_site


@dataclass
class BatchedMemoryState:
    current: list
    pages: list


def batch_states(states):
    """Concatenate per-individual memory tensors without merging their values.

    Every input state must represent exactly one batch row and share the same
    number of sites and history pages. The returned lightweight state is for
    read-only inference; it does not carry clocks, write controls, or provenance.
    """
    states = tuple(states)
    if not states:
        raise ValueError('At least one individual state is required')
    site_count = len(getattr(states[0], 'current', ()))
    page_count = len(getattr(states[0], 'pages', ()))
    if site_count < 1:
        raise ValueError('State must contain current memory sites')
    if any(len(getattr(state, 'current', ())) != site_count
           or len(getattr(state, 'pages', ())) != page_count for state in states):
        raise ValueError('Individuals must have matching sites and history page counts')

    def concatenate(tensors, label):
        first = tensors[0]
        if not isinstance(first, torch.Tensor) or first.ndim < 1 or first.shape[0] != 1:
            raise ValueError(f'{label} must contain one leading row per individual')
        for tensor in tensors:
            if (not isinstance(tensor, torch.Tensor) or tensor.ndim != first.ndim
                    or tensor.shape[0] != 1 or tensor.shape[1:] != first.shape[1:]
                    or tensor.dtype != first.dtype or tensor.device != first.device):
                raise ValueError(f'{label} tensors must have matching shape, dtype and device')
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'{label} tensors must be finite')
        return torch.cat([tensor.detach() for tensor in tensors], dim=0)

    current = [concatenate([state.current[site] for state in states], 'Current memory')
               for site in range(site_count)]
    pages = []
    for page_index in range(page_count):
        page_sites = []
        for site in range(site_count):
            histories = [history_site(state.pages[page_index], site) for state in states]
            width = len(histories[0])
            if width < 1 or any(len(items) != width for items in histories):
                raise ValueError('History page tensors must have matching slots')
            page_sites.append([concatenate([items[slot] for items in histories], 'History memory')
                               for slot in range(width)])
        pages.append(page_sites)
    return BatchedMemoryState(current=current, pages=pages)
