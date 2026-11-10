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
        matrices, addresses, read_mask, memory_slots = active
        address = None if addresses is None else addresses[self.site]
        if read_mask is None and memory_slots is None:
            return self._reader(output, matrices[self.site], self.site, address)
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise ValueError('Layer output must have shape [batch, sequence, hidden_width]')
        batch = output.shape[0]
        if read_mask is not None and read_mask.shape[0] not in (1, batch):
            raise ValueError('Memory read mask must match the decoder batch')
        mask = (torch.ones(batch, dtype=torch.bool, device=output.device) if read_mask is None
                else read_mask.to(device=output.device).expand(batch))
        selected = mask.nonzero(as_tuple=True)[0]
        if not selected.numel():
            return output

        def selected_rows(tensor, name):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1 or tensor.shape[0] not in (1, batch):
                raise ValueError(f'{name} must have a singleton or matching batch')
            return tensor if tensor.shape[0] == 1 else tensor.index_select(0, selected.to(tensor.device))

        site_matrices = matrices[self.site]
        if memory_slots is None:
            selected_matrices = tuple(selected_rows(matrix, 'Memory matrix') for matrix in site_matrices)
        else:
            slots = memory_slots.to(device=output.device).expand(batch).index_select(0, selected)
            if bool(((slots < 0) | (slots >= len(site_matrices))).any()):
                raise ValueError('Memory slot outside the supplied current/history pages')
            expanded_matrices = []
            for matrix in site_matrices:
                if (not isinstance(matrix, torch.Tensor) or matrix.ndim != 3
                        or matrix.shape[0] not in (1, batch)):
                    raise ValueError('Memory matrix must have a singleton or matching batch')
                expanded_matrices.append(matrix.expand(batch, -1, -1))
            stacked_matrices = torch.stack(expanded_matrices, dim=1)
            row_indices = selected.to(stacked_matrices.device)
            row_slots = slots.to(stacked_matrices.device)
            selected_matrix = stacked_matrices[row_indices, row_slots]
            selected_matrices = (selected_matrix,)
        selected_address = None if address is None else selected_rows(address, 'Memory address')
        selected_output = output.index_select(0, selected)
        injected = self._reader(selected_output, selected_matrices, self.site, selected_address)
        return output.index_copy(0, selected, injected)


class IntegratedOpenBase(nn.Module):
    """A language model with two explicit read layers and per-call state.

    Memory is optional so that ordinary base forwards remain possible. The
    reader is applied only when ``memory_state`` is supplied to ``forward`` or
    ``generate``. A model cache must not be reused across memory changes.
    ``memory_read_mask`` optionally selects which batch rows receive the read;
    its boolean vector must have length one or match the decoder batch. The
    caller decides the mask and must account for any generation batch expansion.
    Masked rows skip the reader; state snapshots are still created for the call.
    This is not an access control boundary.
    ``memory_slots`` optionally selects one matrix per row: slot 0 is current
    memory and slots 1..N are history pages in chronological state order. This
    is a caller-supplied temporal address, not inferred from natural language.
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
    def _snapshot(self, state, addresses, current_only, read_mask, memory_slots):
        if type(current_only) is not bool:
            raise ValueError('current_only must be a boolean')
        if self._memory_context.get() is not None:
            raise ValueError('Nested memory calls are not supported')
        if state is None:
            if addresses is not None or read_mask is not None or memory_slots is not None:
                raise ValueError('Addresses and read controls require a memory state')
            yield
            return
        if addresses is not None and (not isinstance(addresses, (tuple, list)) or len(addresses) != 2
                                      or any(not isinstance(address, torch.Tensor) for address in addresses)):
            raise ValueError('Addresses must contain one tensor per site')
        if read_mask is not None and (not isinstance(read_mask, torch.Tensor) or read_mask.dtype != torch.bool
                                      or read_mask.ndim != 1 or not read_mask.numel()):
            raise ValueError('Memory read mask must be a nonempty boolean vector')
        if memory_slots is not None and (not isinstance(memory_slots, torch.Tensor)
                                         or memory_slots.dtype != torch.long or memory_slots.ndim != 1
                                         or not memory_slots.numel()):
            raise ValueError('Memory slots must be a nonempty integer vector')
        with torch.no_grad():
            matrices = tuple(tuple(t.detach().clone() for t in
                                   [state.current[site]] + ([] if current_only else
                                    [history_site(page, site)[0] for page in state.pages]))
                             for site in range(2))
            fixed_addresses = None if addresses is None else tuple(address.detach().clone() for address in addresses)
            fixed_mask = None if read_mask is None else read_mask.detach().clone()
            fixed_slots = None if memory_slots is None else memory_slots.detach().clone()
        token = self._memory_context.set((matrices, fixed_addresses, fixed_mask, fixed_slots))
        try:
            yield
        finally:
            self._memory_context.reset(token)

    def forward(self, *args, memory_state=None, memory_addresses=None, memory_read_mask=None, memory_slots=None,
                current_only=False, **kwargs):
        if memory_state is not None and len(args) > 1:
            raise ValueError('Memory calls accept at most one positional base argument')
        if memory_state is not None and kwargs.get('past_key_values') is not None:
            raise ValueError('External key-value caches are not supported with memory state')
        with self._snapshot(memory_state, memory_addresses, current_only, memory_read_mask, memory_slots):
            return self.base(*args, **kwargs)

    def generate(self, *args, memory_state=None, memory_addresses=None, memory_read_mask=None, memory_slots=None,
                 current_only=False, **kwargs):
        if memory_state is not None and len(args) > 1:
            raise ValueError('Memory calls accept at most one positional base argument')
        if memory_state is not None and kwargs.get('past_key_values') is not None:
            raise ValueError('External key-value caches are not supported with memory state')
        with self._snapshot(memory_state, memory_addresses, current_only, memory_read_mask, memory_slots):
            return self.base.generate(*args, **kwargs)
