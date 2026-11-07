"""Experimental language reader with explicit current/history roles.

History slots encode allocated page order, not semantic event time. Only pages
with observations participate. Frozen sealed factors are read directly; their
precision matrices are unnecessary for this value read. This does not establish
semantic alignment, provenance understanding, or evidence-calibrated retrieval.
"""
import math

import torch
from torch import nn

from .sleep_memory import SealedPage


def frozen_value_read(query, matrix_or_factors):
    """Return M q (or L R q), preserving gradients to the query only."""
    if isinstance(matrix_or_factors, tuple):
        left, right = matrix_or_factors
        return (query @ right.detach().T) @ left.detach().T
    matrix = matrix_or_factors.detach()
    return torch.einsum('bmd,bnd->bnm', matrix.expand(query.shape[0], -1, -1), query)


class TypedMemoryReader(nn.Module):
    def __init__(self, width, sites=2, max_pages=16):
        super().__init__()
        self.max_pages = max_pages
        self.page_keys = nn.Parameter(torch.empty(sites, max_pages, width))
        nn.init.normal_(self.page_keys, std=.02)

    def forward(self, query, state, site, time_count):
        if (type(state.cursor) is not int or not 0 <= state.cursor <= len(state.pages)*time_count
                or not 1 <= len(state.pages) <= self.max_pages):
            raise ValueError('Invalid memory extent for typed reader')
        # Capacity is not evidence: untouched reserved pages must not dilute a read.
        used = (state.cursor + time_count - 1) // time_count
        current = frozen_value_read(query, state.current[site])
        values = []
        for page in state.pages[:used]:
            if isinstance(page, SealedPage):
                left, right, _ = page.factors[site]
                values.append(frozen_value_read(query, (left, right)))
            else:
                values.append(frozen_value_read(query, page[site][0]))
        history = torch.zeros_like(current)
        if values:
            values = torch.stack(values, dim=2)
            keys = self.page_keys[site, :used]
            scores = ((values * query.unsqueeze(2)).sum(-1)
                      + torch.einsum('bnm,pm->bnp', query, keys)) / math.sqrt(query.shape[-1])
            history = (values * scores.softmax(-1).unsqueeze(-1)).sum(2)
        # Concatenation retains the role before the learned output projection.
        # A symmetric sum would erase the distinction even with more parameters.
        return torch.cat((current, history), dim=-1)
