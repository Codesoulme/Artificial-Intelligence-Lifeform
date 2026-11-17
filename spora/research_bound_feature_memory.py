"""Research-only random-feature states bound to a projection and context."""
import hashlib
import json
import torch
from spora import research_random_feature_attention as core


def projection_digest(projection):
    if (not isinstance(projection, torch.Tensor) or projection.ndim != 2
            or projection.dtype != torch.float32 or not bool(torch.isfinite(projection).all())):
        raise ValueError('Projection must be a finite float32 matrix')
    cpu = projection.detach().to('cpu').contiguous()
    header = json.dumps({'shape': list(cpu.shape), 'dtype': 'float32'}, separators=(',', ':')).encode()
    return hashlib.sha256(header + cpu.numpy().tobytes()).hexdigest()


class BoundFeatureMemory:
    """Checks compatible state identity; not a transactional persistence API."""
    def __init__(self, projection, *, base_sha256, layer, kv_head):
        if (not isinstance(base_sha256, str) or len(base_sha256) != 64
                or any(c not in '0123456789abcdef' for c in base_sha256)
                or type(layer) is not int or layer < 0
                or type(kv_head) is not int or kv_head < 0):
            raise ValueError('Invalid base, layer, or head identity')
        self._projection = projection.detach().clone()
        self.identity = {'format': 'spora-feature-memory-v1', 'base_sha256': base_sha256,
                         'layer': layer, 'kv_head': kv_head,
                         'projection_sha256': projection_digest(self._projection)}

    def _check(self, state):
        if (not isinstance(state, dict) or set(state) != {'identity', 'payload'}
                or state['identity'] != self.identity
                or projection_digest(self._projection) != self.identity['projection_sha256']):
            raise ValueError('Memory and projection identity mismatch')
        payload = state['payload']
        if not isinstance(payload, (tuple, list)) or len(payload) != 3:
            raise ValueError('Invalid memory payload')
        matrix, total, shift = payload
        if (not all(isinstance(t, torch.Tensor) and t.dtype == torch.float32 for t in payload)
                or matrix.ndim != 2 or matrix.shape[0] != self._projection.shape[0]
                or matrix.shape[1] < 1 or total.shape != (matrix.shape[0],) or shift.ndim != 0
                or any(t.device != self._projection.device or not bool(torch.isfinite(t).all()) for t in payload)
                or bool((total < 0).any()) or not bool((total > 0).any())):
            raise ValueError('Invalid or nonfinite memory payload')
        return payload

    def write(self, keys, values):
        return {'identity': dict(self.identity), 'payload': core.write(keys, values, self._projection)}

    def merge(self, first, second):
        return {'identity': dict(self.identity), 'payload': core.merge(self._check(first), self._check(second))}

    def read(self, queries, state):
        return core.read(queries, self._projection, self._check(state))
