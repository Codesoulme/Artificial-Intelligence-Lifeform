"""Validated checkpoints for isolated low-rank editing experiments.

These files are not SPORA individual checkpoints and must not be promoted to a
runtime without the relevant behavioral and persistence evaluations.
"""
import copy
from pathlib import Path

import torch

from .language_training import atomic_save


FORMAT = 'spora-research-lowrank-edit-v1'


def _validate_finite(value):
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite optimizer tensor')
    elif isinstance(value, dict):
        for item in value.values():
            _validate_finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _validate_finite(item)


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _adapter_state(modules):
    if not isinstance(modules, (tuple, list)) or not modules:
        raise ValueError('Nonempty low-rank adapter modules are required')
    states = []
    for module in modules:
        if not isinstance(getattr(module, 'down', None), torch.nn.Linear) or not isinstance(
                getattr(module, 'up', None), torch.nn.Linear):
            raise ValueError('Each adapter must expose down and up linear layers')
        state = {'down.weight': module.down.weight, 'up.weight': module.up.weight}
        if any(not bool(torch.isfinite(tensor).all()) for tensor in state.values()):
            raise ValueError('Nonfinite adapter tensor')
        states.append(_cpu_copy(state))
    return states


def _optimizer_parameter_layout(modules, optimizer):
    names = {}
    for index, module in enumerate(modules):
        for name in ('down', 'up'):
            parameter = getattr(module, name).weight
            if id(parameter) in names:
                raise ValueError('Adapter parameters must not be shared')
            names[id(parameter)] = f'{index}.{name}.weight'
    layout = []
    seen = set()
    for group in optimizer.param_groups:
        entries = []
        for parameter in group['params']:
            key = id(parameter)
            if key not in names or key in seen:
                raise ValueError('Optimizer parameters do not match adapters')
            seen.add(key)
            entries.append(names[key])
        layout.append(entries)
    if seen != set(names):
        raise ValueError('Optimizer parameters do not match adapters')
    return layout


def save(path, modules, optimizer, *, base_sha256, feature_sha256, layers, steps):
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    if (not isinstance(base_sha256, str) or not base_sha256
            or not isinstance(feature_sha256, str) or not feature_sha256
            or not isinstance(layers, (tuple, list)) or len(layers) != len(modules)
            or any(type(layer) is not int or layer < 0 for layer in layers)
            or len(set(layers)) != len(layers) or type(steps) is not int or steps < 0
            or not isinstance(optimizer, torch.optim.Optimizer)):
        raise ValueError('Invalid research checkpoint binding or progress')
    _adapter_state(modules)
    layout = _optimizer_parameter_layout(modules, optimizer)
    optimizer_state = _cpu_copy(optimizer.state_dict())
    _validate_finite(optimizer_state)
    payload = {'format': FORMAT, 'base_sha256': base_sha256,
               'feature_sha256': feature_sha256, 'layers': list(layers), 'steps': steps,
               'adapters': _adapter_state(modules), 'optimizer': optimizer_state,
               'optimizer_type': type(optimizer).__module__ + '.' + type(optimizer).__qualname__,
               'optimizer_layout': layout}
    atomic_save(payload, destination)


def restore(path, modules, *, base_sha256, feature_sha256, layers, optimizer=None,
            allow_legacy_optimizer=False):
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if (not isinstance(payload, dict) or payload.get('format') != FORMAT
            or payload.get('base_sha256') != base_sha256
            or payload.get('feature_sha256') != feature_sha256
            or payload.get('layers') != list(layers)):
        raise ValueError('Research checkpoint format or base binding mismatch')
    expected = _adapter_state(modules)
    actual = payload.get('adapters')
    if (not isinstance(actual, list) or len(actual) != len(expected)
            or type(payload.get('steps')) is not int or payload['steps'] < 0):
        raise ValueError('Invalid research checkpoint adapter set or progress')
    for source, reference in zip(actual, expected):
        if not isinstance(source, dict) or set(source) != set(reference):
            raise ValueError('Invalid research checkpoint adapter keys')
        for key, value in source.items():
            if (not isinstance(value, torch.Tensor) or value.shape != reference[key].shape
                    or value.dtype != reference[key].dtype or not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid research checkpoint adapter tensor')
    if optimizer is not None and (not isinstance(optimizer, torch.optim.Optimizer)
                                  or not isinstance(payload.get('optimizer'), dict)):
        raise ValueError('Invalid research checkpoint optimizer')
    if optimizer is not None:
        optimizer_type = type(optimizer).__module__ + '.' + type(optimizer).__qualname__
        if payload.get('optimizer_type') != optimizer_type:
            raise ValueError('Research checkpoint optimizer type mismatch')
        layout = _optimizer_parameter_layout(modules, optimizer)
        saved_layout = payload.get('optimizer_layout')
        if saved_layout is None and not allow_legacy_optimizer:
            raise ValueError('Legacy optimizer layout requires explicit verification')
        if saved_layout is not None and saved_layout != layout:
            raise ValueError('Research checkpoint optimizer parameter order mismatch')
        _validate_finite(payload['optimizer'])
        # Validate parameter groups before changing any live state.
        copy.deepcopy(optimizer).load_state_dict(copy.deepcopy(payload['optimizer']))
    old_optimizer = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    try:
        if optimizer is not None:
            optimizer.load_state_dict(payload['optimizer'])
        for module, state in zip(modules, actual):
            module.down.load_state_dict({'weight': state['down.weight']}, strict=True)
            module.up.load_state_dict({'weight': state['up.weight']}, strict=True)
    except Exception:
        for module, state in zip(modules, expected):
            with torch.no_grad():
                module.down.weight.copy_(state['down.weight'])
                module.up.weight.copy_(state['up.weight'])
        if optimizer is not None:
            optimizer.load_state_dict(old_optimizer)
        raise
    return {'steps': payload['steps'], 'layers': payload['layers']}
