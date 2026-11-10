"""SPORA-native causal block for layer-by-layer teacher distillation."""
import torch
from torch import nn

from .model import CausalBlock, ModelConfig


class SPORALayerReplacement(nn.Module):
    """A distilled SPORA causal block with learned absolute position input.

    It accepts the Qwen2 decoder-layer call shape for full-sequence evaluation.
    Autoregressive KV caching is deliberately rejected until an incremental
    cache implementation is added and parity-tested.
    """

    attention_type = 'full_attention'

    def __init__(self, width, heads, max_length=512, layer_idx=None):
        super().__init__()
        if (type(width) is not int or width <= 0 or type(heads) is not int or heads <= 0
                or width % heads or type(max_length) is not int or max_length <= 0):
            raise ValueError('Invalid SPORA replacement dimensions')
        config = ModelConfig(width=width, heads=heads, max_length=max_length)
        config.validate()
        self.width, self.max_length = width, max_length
        if layer_idx is not None and (type(layer_idx) is not int or layer_idx < 0):
            raise ValueError('Layer index must be a nonnegative integer')
        self.layer_idx = layer_idx
        self.position = nn.Embedding(max_length, width)
        self.block = CausalBlock(config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        if use_cache or past_key_values is not None:
            return self._cached_forward(hidden_states, attention_mask, position_ids,
                                        past_key_values, use_cache, cache_position)
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3 or hidden_states.shape[-1] != self.width:
            raise ValueError('Expected hidden states with shape [batch, sequence, width]')
        batch, length = hidden_states.shape[:2]
        if not length or length > self.max_length:
            raise ValueError('Sequence length outside replacement position range')
        if position_ids is None:
            position_ids = torch.arange(length, device=hidden_states.device).expand(batch, -1)
        if (position_ids.ndim != 2 or position_ids.shape[0] not in (1, batch)
                or position_ids.shape[1] != length or position_ids.dtype != torch.long
                or bool(((position_ids < 0) | (position_ids >= self.max_length)).any())):
            raise ValueError('Invalid absolute position IDs')
        position_ids = position_ids.expand(batch, -1)
        hidden_dtype = hidden_states.dtype
        positions = self.position(position_ids.to(self.position.weight.device))
        values = hidden_states.to(device=positions.device, dtype=positions.dtype) + positions
        return self.block(values).to(device=hidden_states.device, dtype=hidden_dtype)

    def _cached_forward(self, hidden_states, attention_mask, position_ids,
                        past_key_values, use_cache, cache_position):
        if not use_cache or past_key_values is None:
            raise ValueError('Cached calls require use_cache=True and a cache object')
        if self.layer_idx is None or not callable(getattr(past_key_values, 'update', None)):
            raise ValueError('Cached calls require a configured layer index and mutable Cache')
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3 or hidden_states.shape[-1] != self.width:
            raise ValueError('Expected hidden states with shape [batch, sequence, width]')
        batch, length = hidden_states.shape[:2]
        if not length or length > self.max_length:
            raise ValueError('Sequence length outside replacement position range')
        if position_ids is None:
            previous = int(past_key_values.get_seq_length(self.layer_idx))
            position_ids = torch.arange(previous, previous + length, device=hidden_states.device).expand(batch, -1)
        if (position_ids.ndim != 2 or position_ids.shape[0] not in (1, batch)
                or position_ids.shape[1] != length or position_ids.dtype != torch.long
                or bool(((position_ids < 0) | (position_ids >= self.max_length)).any())):
            raise ValueError('Invalid absolute position IDs')
        position_ids = position_ids.expand(batch, -1)
        past_length = int(past_key_values.get_seq_length(self.layer_idx))
        if past_length and length != 1:
            raise ValueError('Cached continuation currently accepts exactly one new token')
        position = self.position(position_ids.to(self.position.weight.device))
        values = hidden_states.to(device=position.device, dtype=position.dtype) + position
        b, n, d = values.shape
        head_width = d // self.block.heads
        qkv = self.block.qkv(self.block.norm1(values)).reshape(
            b, n, 3, self.block.heads, head_width).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        cache_kwargs = {} if cache_position is None else {'cache_position': cache_position}
        key, value = past_key_values.update(key, value, self.layer_idx, cache_kwargs)
        mask = attention_mask
        if mask is not None:
            if mask.shape[-1] != key.shape[-2] or mask.shape[-2] not in (1, n):
                raise ValueError('Cached attention mask has incompatible query/key lengths')
            if mask.is_floating_point():
                mask = mask.to(device=query.device, dtype=query.dtype)
            else:
                mask = mask.to(device=query.device)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask,
            is_causal=mask is None and past_length == 0 and n > 1)
        attended = attended.transpose(1, 2).reshape(b, n, d)
        values = values + self.block.out(attended)
        values = values + self.block.ff(self.block.norm2(values))
        return values.to(device=hidden_states.device, dtype=hidden_states.dtype)


class SPORALayerBlend(nn.Module):
    """Full-sequence progressive graft of a SPORA block into a teacher layer.

    The scalar blend is an explicit structural interpolation control. KV cache
    is rejected because both branches would otherwise mutate the same cache.
    """

    def __init__(self, teacher_layer, spora_layer, strength):
        super().__init__()
        if not isinstance(teacher_layer, nn.Module) or not isinstance(spora_layer, SPORALayerReplacement):
            raise ValueError('Expected a teacher decoder layer and SPORA replacement')
        if (isinstance(strength, bool) or not isinstance(strength, (float, int))
                or not 0.0 <= float(strength) <= 1.0):
            raise ValueError('Blend strength must be in [0, 1]')
        self.teacher_layer = teacher_layer
        self.spora_layer = spora_layer
        self.strength = float(strength)
        self.attention_type = getattr(teacher_layer, 'attention_type', 'full_attention')

    def forward(self, hidden_states, *args, **kwargs):
        if kwargs.get('use_cache', False) or kwargs.get('past_key_values') is not None:
            raise ValueError('Blended SPORA layers require cache-disabled full-sequence evaluation')
        teacher = self.teacher_layer(hidden_states, *args, **kwargs)
        if not isinstance(teacher, torch.Tensor):
            raise ValueError('Teacher decoder layer must return a tensor')
        replacement_kwargs = dict(kwargs)
        replacement_kwargs['use_cache'] = False
        replacement_kwargs['past_key_values'] = None
        spora = self.spora_layer(hidden_states, *args, **replacement_kwargs)
        return teacher + self.strength * (spora.to(device=teacher.device, dtype=teacher.dtype) - teacher)
