"""Protected parameter shadow for controlled damage and recovery experiments."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch


def state_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode('utf-8'))
        tensor = state[name].detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class DnaShadow:
    """EMA shadow kept separate from active parameters and never auto-applied."""

    def __init__(self, tensors, alpha=.999, steps=0):
        if not 0 < alpha < 1:
            raise ValueError('Alpha must be between zero and one')
        if not tensors:
            raise ValueError('At least one tensor required')
        if any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in tensors.items()):
            raise ValueError('Named tensor state required')
        if not all(bool(torch.isfinite(v).all()) for v in tensors.values()):
            raise ValueError('Nonfinite DNA state')
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.tensors = {k: v.detach().cpu().clone() for k, v in tensors.items()}

    def update(self, active):
        if set(active) != set(self.tensors):
            raise ValueError('DNA and active tensor names differ')
        for name, value in active.items():
            value = value.detach().cpu()
            if value.shape != self.tensors[name].shape or value.dtype != self.tensors[name].dtype:
                raise ValueError('DNA tensor shape or dtype changed: '+name)
            if not bool(torch.isfinite(value).all()):
                raise ValueError('Nonfinite active tensor')
            self.tensors[name].mul_(self.alpha).add_(value, alpha=1-self.alpha)
        self.steps += 1

    def repair_damaged(self, active, threshold=1e-6):
        if threshold <= 0 or not torch.isfinite(torch.tensor(threshold)):
            raise ValueError('Positive finite damage threshold required')
        if set(active) != set(self.tensors):
            raise ValueError('DNA and active tensor names differ')
        repaired = {}
        damaged = {}
        for name, value in active.items():
            if value.shape != self.tensors[name].shape or value.dtype != self.tensors[name].dtype:
                raise ValueError('DNA tensor shape or dtype changed: '+name)
            mask = value.detach().abs() < threshold
            repaired[name] = torch.where(mask.cpu(), self.tensors[name], value.detach().cpu())
            damaged[name] = int(mask.sum())
        return repaired, damaged

    def snapshot(self):
        return {'format': 'spora-dna-shadow-v1', 'alpha': self.alpha,
                'steps': self.steps, 'tensors': {k: v.clone() for k, v in self.tensors.items()},
                'digest': state_digest(self.tensors)}

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.tmp')
        torch.save(self.snapshot(), temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('format') != 'spora-dna-shadow-v1' or state_digest(payload['tensors']) != payload.get('digest'):
            raise ValueError('Invalid or corrupted DNA shadow')
        shadow = cls(payload['tensors'], payload['alpha'], payload['steps'])
        return shadow
