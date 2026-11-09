"""Read-only SPORA memory adapters for tensor-output base-model layers.

``memory_hooks`` snapshots the supplied state on entry and removes both hooks
on exit, including exceptional exit. Re-enter with a new state to switch it.
The caller owns the adapter, base model, and any base-model freezing policy.
Do not use overlapping hook contexts on the same layers or concurrent forwards
that require different states.
"""
from contextlib import contextmanager
from threading import Lock
from weakref import WeakSet

import torch
from torch import nn
from torch.nn import functional as F

from .sleep_memory import history_site

_ACTIVE_LAYERS = WeakSet()
_ACTIVE_LOCK = Lock()


class OpenBaseMemory(nn.Module):
    """Two independent legacy memory readers, one for each supplied layer."""

    def __init__(self, hidden_width, memory_width):
        super().__init__()
        if any(type(n) is not int or n <= 0 for n in (hidden_width, memory_width)):
            raise ValueError('Widths must be positive integers')
        self.hidden_width = hidden_width
        self.memory_width = memory_width
        self.query = nn.ModuleList(nn.Linear(hidden_width, memory_width, bias=False) for _ in range(2))
        self.out = nn.ModuleList(nn.Linear(memory_width, hidden_width, bias=False) for _ in range(2))
        self.gate = nn.Parameter(torch.full((2,), -3.))
        for projection in self.out:
            nn.init.zeros_(projection.weight)

    def forward(self, hidden, matrices, site, address=None):
        if type(site) is not int or site not in (0, 1):
            raise ValueError('Site must be 0 or 1')
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[-1] != self.hidden_width:
            raise ValueError('Layer output must have shape [batch, sequence, hidden_width]')
        if not hidden.is_floating_point() or not hidden.shape[0]:
            raise ValueError('Layer output must be floating point with a nonempty batch')
        if not matrices:
            raise ValueError('At least one memory matrix is required')
        projection = self.query[site]
        x = hidden.to(device=projection.weight.device, dtype=projection.weight.dtype)
        if address is None:
            q = F.normalize(projection(x), dim=-1)
        else:
            if (not isinstance(address, torch.Tensor) or address.ndim != 2
                    or address.shape[1] != self.memory_width or address.shape[0] not in (1, hidden.shape[0])
                    or not address.is_floating_point() or not bool(torch.isfinite(address).all())):
                raise ValueError('Address must have shape [1 or batch, memory_width]')
            q = F.normalize(address.detach().to(device=x.device, dtype=x.dtype), dim=-1)
            q = q.expand(hidden.shape[0], -1).unsqueeze(1).expand(-1, hidden.shape[1], -1)
        values = []
        for matrix in matrices:
            if (not isinstance(matrix, torch.Tensor) or matrix.ndim != 3
                    or matrix.shape[1:] != (self.memory_width, self.memory_width)
                    or matrix.shape[0] not in (1, hidden.shape[0])
                    or not matrix.is_floating_point()):
                raise ValueError('Memory matrices must have shape [1 or batch, memory_width, memory_width]')
            matrix = matrix.detach().to(device=q.device, dtype=q.dtype)
            values.append(torch.einsum('bmd,bnd->bnm', matrix.expand(hidden.shape[0], -1, -1), q))
        values = torch.stack(values, dim=2)
        weights = (values * q.unsqueeze(2)).sum(-1).softmax(-1)
        read = (values * weights.unsqueeze(-1)).sum(2)
        residual = self.gate[site].sigmoid() * self.out[site](read)
        return hidden + residual.to(device=hidden.device, dtype=hidden.dtype)


@contextmanager
def memory_hooks(layers, adapter, state, *, current_only=False, addresses=None):
    """Attach two hooks for one immutable state snapshot and then remove them.

    Layers must be distinct modules whose outputs are tensors of shape
    [batch, sequence, hidden_width]. State matrices may have batch size one
    (broadcast) or match each forward's batch. The state is never mutated.
    """
    layers = tuple(layers)
    if len(layers) != 2 or layers[0] is layers[1] or not all(isinstance(layer, nn.Module) for layer in layers):
        raise ValueError('Exactly two distinct layers are required')
    if not isinstance(adapter, OpenBaseMemory):
        raise ValueError('Expected an OpenBaseMemory adapter')
    if type(current_only) is not bool:
        raise ValueError('current_only must be a boolean')
    if addresses is not None and (not isinstance(addresses, (tuple, list)) or len(addresses) != 2
                                  or any(not isinstance(address, torch.Tensor) for address in addresses)):
        raise ValueError('Addresses must contain one tensor per site')
    handles = []
    with _ACTIVE_LOCK:
        if any(layer in _ACTIVE_LAYERS for layer in layers):
            raise ValueError('Overlapping memory hook contexts are not supported')
        for layer in layers:
            _ACTIVE_LAYERS.add(layer)
    try:
        with torch.no_grad():
            snapshots = tuple(
                tuple(matrix.detach().clone() for matrix in
                      [state.current[site]] + ([] if current_only else [history_site(page, site)[0] for page in state.pages]))
                for site in range(2)
            )
            fixed_addresses = None if addresses is None else tuple(address.detach().clone() for address in addresses)
        for site, layer in enumerate(layers):
            def hook(_module, _inputs, output, site=site):
                return adapter(output, snapshots[site], site, None if fixed_addresses is None else fixed_addresses[site])
            handles.append(layer.register_forward_hook(hook))
        yield adapter
    finally:
        for handle in handles:
            handle.remove()
        with _ACTIVE_LOCK:
            for layer in layers:
                _ACTIVE_LAYERS.discard(layer)
