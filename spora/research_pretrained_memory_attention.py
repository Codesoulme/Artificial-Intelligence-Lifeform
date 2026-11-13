"""Research-only memory attention using a frozen language attention as a prior.

The external codebook and supplied address are diagnostic controls, not a
learned memory router, and no text is appended to the user question.
"""
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F

from .research_oracle_bridge import context_oracle_embeddings


@contextmanager
def disable_memory_attention(module):
    if not isinstance(module, PretrainedMemoryAttention):
        raise ValueError('Expected pretrained memory attention')
    saved = module.gain.detach().clone()
    try:
        with torch.no_grad():
            module.gain.zero_()
        yield
    finally:
        with torch.no_grad():
            module.gain.copy_(saved)


class PretrainedMemoryAttention(nn.Module):
    """Add query-dependent memory attention alongside an existing attention layer."""

    def __init__(self, attention, norm, decoder, codec, embedding, *, slots=4):
        super().__init__()
        if type(slots) is not int or not 1 <= slots <= codec.slot_count:
            raise ValueError('Invalid memory slot count')
        self.inner = attention
        object.__setattr__(self, '_norm', norm)
        object.__setattr__(self, '_decoder', decoder)
        object.__setattr__(self, '_codec', codec)
        object.__setattr__(self, '_embedding', embedding)
        object.__setattr__(self, '_rotary_emb', getattr(getattr(getattr(decoder, 'base', None), 'model', None),
                                                        'rotary_emb', None))
        self.slots = slots
        self.head_dim = attention.head_dim
        self.use_memory_rope = False
        q = getattr(attention.q_proj, 'base', attention.q_proj)
        k = attention.k_proj
        v = getattr(attention.v_proj, 'base', attention.v_proj)
        out = getattr(attention.o_proj, 'base', attention.o_proj)
        if not all(isinstance(x, nn.Linear) for x in (q, k, v, out)):
            raise ValueError('Expected linear attention projections')
        if (q.out_features % self.head_dim or k.out_features % self.head_dim
                or v.out_features != k.out_features or q.in_features != embedding.shape[1]):
            raise ValueError('Unsupported attention head dimensions')
        self.query_heads = q.out_features // self.head_dim
        self.memory_heads = k.out_features // self.head_dim
        if self.query_heads % self.memory_heads:
            raise ValueError('Memory heads must divide query heads')
        self.memory_key = nn.Linear(k.in_features, k.out_features, bias=k.bias is not None,
                                    device=k.weight.device, dtype=torch.float32)
        self.memory_value = nn.Linear(v.in_features, v.out_features, bias=v.bias is not None,
                                      device=v.weight.device, dtype=torch.float32)
        self.memory_key.load_state_dict({n: t.detach().float() for n, t in k.state_dict().items()})
        self.memory_value.load_state_dict({n: t.detach().float() for n, t in v.state_dict().items()})
        self.gain = nn.Parameter(torch.zeros((), device=k.weight.device, dtype=torch.float32))

    def forward(self, *args, **kwargs):
        result = self.inner(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError('Expected attention output and weights')
        active = self._decoder._memory_context.get()
        if active is None or (not torch.is_grad_enabled() and float(self.gain.detach()) == 0.):
            return result
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError('Research memory attention supports a single input row')
        dummy = hidden.new_zeros((1, self.slots, self._codec.slot_width))
        memory = context_oracle_embeddings(self._decoder, self._codec, self._embedding, dummy)
        valid = memory.norm(dim=-1) > 0
        if not bool(valid.any()):
            return result
        memory = self._norm(memory.to(hidden.dtype)).float()
        qproj = getattr(self.inner.q_proj, 'base', self.inner.q_proj)
        oproj = getattr(self.inner.o_proj, 'base', self.inner.o_proj)
        query = qproj(hidden).view(1, hidden.shape[1], self.query_heads, self.head_dim).transpose(1, 2).float()
        key = self.memory_key(memory).view(1, self.slots, self.memory_heads, self.head_dim).transpose(1, 2)
        value = self.memory_value(memory).view(1, self.slots, self.memory_heads, self.head_dim).transpose(1, 2)
        position_embeddings = kwargs.get('position_embeddings')
        if self.use_memory_rope:
            cos = sin = None
            if self._rotary_emb is not None:
                position_ids = torch.arange(self.slots, device=key.device, dtype=torch.long).unsqueeze(0)
                cos, sin = self._rotary_emb(key, position_ids)
            elif position_embeddings is not None:
                cos, sin = position_embeddings
                if (not isinstance(cos, torch.Tensor) or not isinstance(sin, torch.Tensor)
                        or cos.ndim != 3 or sin.shape != cos.shape or cos.shape[1] < self.slots
                        or cos.shape[-1] < self.head_dim):
                    raise ValueError('Memory rotary embeddings have an incompatible shape')
            if cos is None or sin is None or cos.shape[-1] < self.head_dim:
                raise ValueError('Memory rotary embeddings are unavailable')
            if cos.ndim != 3 or sin.shape != cos.shape or cos.shape[1] < self.slots:
                raise ValueError('Memory rotary embeddings have an incompatible shape')
            cos = cos[:, :self.slots, :self.head_dim].to(key.device, key.dtype).unsqueeze(1)
            sin = sin[:, :self.slots, :self.head_dim].to(key.device, key.dtype).unsqueeze(1)
            rotated = key * cos + torch.cat((-key[..., self.head_dim // 2:],
                                              key[..., :self.head_dim // 2]), dim=-1) * sin
            key = rotated
        repeats = self.query_heads // self.memory_heads
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(~valid[:, None, None, :], float('-inf'))
        output = torch.matmul(scores.softmax(dim=-1), value).transpose(1, 2).reshape(1, hidden.shape[1], -1)
        delta = oproj(output.to(oproj.weight.dtype)).to(result[0].dtype)
        return result[0] + self.gain.to(delta.dtype) * delta, result[1]
