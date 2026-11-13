"""Research-only test-time memory graft based on next-state targets.

This keeps the host decoder's native MLP and stores only per-individual fast
weight deltas outside the checkpoint. The write target is a contextual hidden
state at the next position, rather than a finite symbolic value embedding.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn
from torch.nn import functional as F


_ACTIVE_MEMORY = ContextVar('spora_ttt_ntp_memory', default=None)


class TTTNTPMemoryLinear(nn.Module):
    """A frozen linear layer with an optional per-call fast-weight delta."""

    def __init__(self, base, layer_id):
        super().__init__()
        if not isinstance(base, nn.Linear) or type(layer_id) is not int or layer_id < 0:
            raise ValueError('A linear base and nonnegative layer id are required')
        self.base = base
        self.base.requires_grad_(False)
        self.layer_id = layer_id
        self.read_adapter = None

    def forward(self, hidden):
        output = self.base(hidden)
        active = _ACTIVE_MEMORY.get()
        if active is None or self.layer_id not in active:
            return output
        fast = active[self.layer_id]
        if (not isinstance(fast, torch.Tensor) or fast.ndim != 2
                or tuple(fast.shape) != tuple(self.base.weight.shape)
                or not fast.is_floating_point() or not bool(torch.isfinite(fast).all())):
            raise ValueError('Fast weight has an incompatible shape or value')
        delta = F.linear(hidden.float(), fast.float())
        if self.read_adapter is not None:
            return output + self.read_adapter(delta).to(device=output.device, dtype=output.dtype)
        return output + delta.to(device=output.device, dtype=output.dtype)


class TTTNTPReadAdapter(nn.Module):
    """Low-rank learned map from a TTT read vector to a decoder residual."""

    def __init__(self, width, rank=16):
        super().__init__()
        if type(width) is not int or width < 2 or type(rank) is not int or not 1 <= rank <= width:
            raise ValueError('Invalid TTT read adapter dimensions')
        self.down = nn.Linear(width, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, width, bias=False, dtype=torch.float32)
        nn.init.normal_(self.down.weight, std=width ** -0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, read):
        if not isinstance(read, torch.Tensor) or read.shape[-1] != self.down.in_features:
            raise ValueError('TTT read has an incompatible hidden width')
        return self.up(torch.tanh(self.down(read.float())))


def install(model, layer_ids=(6, 12)):
    """Wrap Qwen-style MLP down projections and return the wrappers."""
    layers = getattr(getattr(model, 'model', None), 'layers', None)
    if not isinstance(layers, nn.ModuleList) or not layer_ids:
        raise ValueError('A decoder with model.layers is required')
    wrappers = {}
    for layer_id in layer_ids:
        if type(layer_id) is not int or not 0 <= layer_id < len(layers):
            raise ValueError('Layer id is outside the decoder')
        mlp = getattr(layers[layer_id], 'mlp', None)
        down = getattr(mlp, 'down_proj', None)
        if isinstance(down, TTTNTPMemoryLinear) or not isinstance(down, nn.Linear):
            raise ValueError('Expected an unwrapped linear down projection')
        wrapper = TTTNTPMemoryLinear(down, layer_id)
        mlp.down_proj = wrapper
        wrappers[layer_id] = wrapper
    return wrappers


def fresh_memory(wrappers, *, dtype=torch.float32):
    """Create zero fast weights matching the installed down projections."""
    if not isinstance(wrappers, dict) or not wrappers:
        raise ValueError('Installed wrappers are required')
    result = {}
    for layer_id, wrapper in wrappers.items():
        result[layer_id] = torch.zeros_like(wrapper.base.weight, dtype=dtype,
                                             device=wrapper.base.weight.device)
    return result


def attach_read_adapters(wrappers, *, rank=16):
    """Attach zero-output low-rank adapters while preserving the base path."""
    if not isinstance(wrappers, dict) or not wrappers:
        raise ValueError('Installed wrappers are required')
    for wrapper in wrappers.values():
        if not isinstance(wrapper, TTTNTPMemoryLinear) or wrapper.read_adapter is not None:
            raise ValueError('Expected unmodified TTT wrappers')
        wrapper.read_adapter = TTTNTPReadAdapter(wrapper.base.out_features, rank=rank).to(
            device=wrapper.base.weight.device)
    return {layer_id: wrapper.read_adapter for layer_id, wrapper in wrappers.items()}


@contextmanager
def use_memory(memory):
    """Expose an immutable snapshot to the wrapped layers for one call."""
    if not isinstance(memory, dict) or any(not isinstance(key, int) for key in memory):
        raise ValueError('Memory must be a layer-indexed dictionary')
    snapshot = {key: value.detach() for key, value in memory.items()}
    token = _ACTIVE_MEMORY.set(snapshot)
    try:
        yield
    finally:
        _ACTIVE_MEMORY.reset(token)


@torch.no_grad()
def sequential_ntp_write(memory, keys, targets, *, eta=0.1, ridge=1.0, max_steps=None):
    """Apply rank-one next-state writes to every selected layer.

    ``keys`` may be a shared tensor or a layer-indexed dictionary of tensors;
    each tensor has shape ``[tokens, input_width]``. ``targets`` is a
    layer-indexed dictionary of ``[tokens, output_width]`` tensors. Each update
    sees the fast weight produced by earlier positions, preserving causal
    online-write behavior.
    """
    valid_keys = isinstance(keys, torch.Tensor) or (isinstance(keys, dict) and bool(keys))
    if (not isinstance(memory, dict) or not memory or not valid_keys
            or not isinstance(targets, dict)
            or not 0 < eta <= 1 or not 0 < ridge or max_steps is not None and max_steps < 1):
        raise ValueError('Invalid NTP write inputs')
    for layer_id, fast in memory.items():
        layer_keys = keys.get(layer_id) if isinstance(keys, dict) else keys
        target = targets.get(layer_id)
        if (not isinstance(layer_keys, torch.Tensor) or layer_keys.ndim != 2
                or not layer_keys.shape[0] or not isinstance(target, torch.Tensor) or target.ndim != 2
                or target.shape[0] != layer_keys.shape[0]
                or target.shape[1] != fast.shape[0]
                or layer_keys.shape[1] != fast.shape[1]):
            raise ValueError('Layer key/target dimensions do not match fast weight')
        steps = layer_keys.shape[0]
        if max_steps is not None:
            steps = min(steps, max_steps)
        for position in range(steps):
            key = layer_keys[position].float()
            value = target[position].float()
            prediction = F.linear(key.unsqueeze(0), fast.float())[0]
            residual = value - prediction
            denominator = key.square().sum() + ridge
            fast.add_(eta * residual.unsqueeze(1) * key.unsqueeze(0) / denominator)
        if not bool(torch.isfinite(fast).all()):
            raise ValueError('NTP write produced non-finite fast weights')
    return memory
