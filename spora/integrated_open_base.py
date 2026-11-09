"""Experimental decoder-layer integration of a SPORA memory reader.

Construction wraps two layers of the supplied base model in place. The wrapper
owns the reader and passes immutable, per-call memory snapshots via a context
variable; no forward hooks are installed. The caller owns persistence, routing,
and any concurrency policy for the underlying language model.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import math

import torch
from torch import nn

from .open_base_memory import OpenBaseMemory
from .sleep_memory import history_site


class ScaledOpenBaseMemory(OpenBaseMemory):
    """Reader whose residual is scaled after the base dtype conversion."""

    def __init__(self, hidden_width, memory_width, strength):
        super().__init__(hidden_width, memory_width)
        if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('Strength must be finite and in [0, 1]')
        self.strength = float(strength)

    def forward(self, hidden, matrices, site, address=None):
        injected = super().forward(hidden, matrices, site, address)
        return hidden + self.strength * (injected - hidden)


class _MemoryDecoderLayer(nn.Module):
    def __init__(self, layer, reader, site, context):
        super().__init__()
        self.layer = layer
        self.attention_type = layer.attention_type
        self.site = site
        object.__setattr__(self, '_reader', reader)
        object.__setattr__(self, '_memory_context', context)

    def forward(self, *args, **kwargs):
        output = self.layer(*args, **kwargs)
        active = self._memory_context.get()
        if active is None:
            return output
        matrices, addresses = active
        address = None if addresses is None else addresses[self.site]
        return self._reader(output, matrices[self.site], self.site, address)


class IntegratedOpenBase(nn.Module):
    """A language model with two explicit read layers and per-call state.

    Memory is optional so that ordinary base forwards remain possible. The
    reader is applied only when ``memory_state`` is supplied to ``forward`` or
    ``generate``. A model cache must not be reused across memory changes.
    """

    def __init__(self, base, reader, layer_indices):
        super().__init__()
        if not isinstance(base, nn.Module) or not isinstance(reader, OpenBaseMemory):
            raise ValueError('Expected a base module and an OpenBaseMemory reader')
        layers = getattr(getattr(base, 'model', None), 'layers', None)
        if not isinstance(layers, nn.ModuleList):
            raise ValueError('Expected base.model.layers as a ModuleList')
        if (not isinstance(layer_indices, (tuple, list)) or len(layer_indices) != 2
                or any(type(i) is not int or not 0 <= i < len(layers) for i in layer_indices)
                or layer_indices[0] == layer_indices[1]):
            raise ValueError('Expected two distinct layer indices')
        selected = [layers[i] for i in layer_indices]
        if any(isinstance(layer, _MemoryDecoderLayer) or not hasattr(layer, 'attention_type') for layer in selected):
            raise ValueError('Layers must be unwrapped tensor-output decoder layers')
        self.base = base
        self.reader = reader
        self.layer_indices = tuple(layer_indices)
        self._memory_context = ContextVar('spora_integrated_memory', default=None)
        for site, index in enumerate(self.layer_indices):
            layers[index] = _MemoryDecoderLayer(layers[index], reader, site, self._memory_context)

    @contextmanager
    def _snapshot(self, state, addresses, current_only):
        if type(current_only) is not bool:
            raise ValueError('current_only must be a boolean')
        if self._memory_context.get() is not None:
            raise ValueError('Nested memory calls are not supported')
        if state is None:
            if addresses is not None:
                raise ValueError('Addresses require a memory state')
            yield
            return
        if addresses is not None and (not isinstance(addresses, (tuple, list)) or len(addresses) != 2
                                      or any(not isinstance(address, torch.Tensor) for address in addresses)):
            raise ValueError('Addresses must contain one tensor per site')
        with torch.no_grad():
            matrices = tuple(tuple(t.detach().clone() for t in
                                   [state.current[site]] + ([] if current_only else
                                    [history_site(page, site)[0] for page in state.pages]))
                             for site in range(2))
            fixed_addresses = None if addresses is None else tuple(address.detach().clone() for address in addresses)
        token = self._memory_context.set((matrices, fixed_addresses))
        try:
            yield
        finally:
            self._memory_context.reset(token)

    def forward(self, *args, memory_state=None, memory_addresses=None, current_only=False, **kwargs):
        if memory_state is not None and len(args) > 1:
            raise ValueError('Memory calls accept at most one positional base argument')
        if memory_state is not None and kwargs.get('past_key_values') is not None:
            raise ValueError('External key-value caches are not supported with memory state')
        with self._snapshot(memory_state, memory_addresses, current_only):
            return self.base(*args, **kwargs)

    def generate(self, *args, memory_state=None, memory_addresses=None, current_only=False, **kwargs):
        if memory_state is not None and len(args) > 1:
            raise ValueError('Memory calls accept at most one positional base argument')
        if memory_state is not None and kwargs.get('past_key_values') is not None:
            raise ValueError('External key-value caches are not supported with memory state')
        with self._snapshot(memory_state, memory_addresses, current_only):
            return self.base.generate(*args, **kwargs)
