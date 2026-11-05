"""Atomic neural checkpoints. No raw experiences or lookup answers."""
from pathlib import Path
import os
import torch
from .model import SPORA, ModelConfig


def _cpu_state(nested):
    return [[p.detach().cpu().clone() for p in site] for site in nested]


def save_checkpoint(path, model, fast=None, velocity=None, cursor=0, metadata=None,
                    optimizer=None, training_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if (fast is None) != (velocity is None):
        raise ValueError('Fast parameters and optimizer momentum must be saved together')
    tensors = list(model.state_dict().values())
    if fast is not None:
        tensors += [p for site in fast+velocity for p in site]
    if not all(bool(torch.isfinite(p).all()) for p in tensors):
        raise ValueError('Refusing nonfinite checkpoint')
    data = {
        'format_version': 1, 'config': model.config_dict(),
        'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'fast': None if fast is None else _cpu_state(fast),
        'velocity': None if velocity is None else _cpu_state(velocity),
        'cursor': int(cursor), 'metadata': metadata or {},
    }
    if optimizer is not None:
        data['outer_optimizer'] = optimizer.state_dict()
    if training_state is not None:
        data['training_state'] = training_state
    temporary = path.with_suffix(path.suffix+'.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path, device='cpu', reset_fast=False):
    data = torch.load(Path(path), map_location='cpu', weights_only=True)
    if data.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint schema')
    if 'temporal_kind' in data['config']:
        from .temporal import TemporalSPORA,TemporalConfig
        model=TemporalSPORA(TemporalConfig(**data['config'])).to(device)
        if type(data['cursor']) is not int or not 0<=data['cursor']<=model.config.time_count:
            raise ValueError('Invalid temporal event cursor')
    else:
        model = SPORA(ModelConfig(**data['config'])).to(device)
    model.load_state_dict(data['model'], strict=True)
    model.eval()
    if data['fast'] is None or reset_fast:
        batch_size = 1 if data['fast'] is None else data['fast'][0][0].shape[0]
        fast, velocity = model.initial_state(batch_size)
    else:
        fast = [[p.to(device).detach().requires_grad_(True) for p in s] for s in data['fast']]
        velocity = [[p.to(device) for p in s] for s in data['velocity']]
        if len(fast) != len(model.memories) or len(velocity) != len(fast):
            raise ValueError('Memory attachment mismatch')
        batch_size=fast[0][0].shape[0]
        for params, moments, site in zip(fast, velocity, model.memories):
            if len(params) != len(site.initial) or len(moments) != len(params):
                raise ValueError('Incomplete memory state')
            for p, u, init in zip(params, moments, site.initial):
                if p.shape[1:] != init.shape or p.shape != u.shape or p.shape[0] != batch_size:
                    raise ValueError('Memory tensor shape mismatch')
    if not all(bool(torch.isfinite(p).all()) for p in list(model.state_dict().values()) +
               [p for s in fast+velocity for p in s]):
        raise ValueError('Nonfinite checkpoint tensors')
    return model, fast, velocity, data
