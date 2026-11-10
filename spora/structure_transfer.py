"""SPORA-native causal block for layer-by-layer teacher distillation."""
import torch
from torch import nn

from .model import CausalBlock, ModelConfig


def _attention_mask(mask, batch, heads, queries, keys, past, device, dtype):
    """Convert padding or additive attention masks and enforce causality."""
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise ValueError('Attention mask must be a tensor')
    if mask.ndim == 2:
        if mask.shape != (batch, keys) or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError('Invalid 2-D padding attention mask')
        allowed = mask.to(device=device, dtype=torch.bool)[:, None, None, :]
        additive = torch.zeros((batch, 1, 1, keys), device=device, dtype=dtype)
        additive = additive.masked_fill(~allowed, float('-inf'))
    elif mask.ndim == 4:
        if (mask.shape[0] not in (1, batch) or mask.shape[1] not in (1, heads)
                or mask.shape[2] not in (1, queries) or mask.shape[3] != keys):
            raise ValueError('Invalid 4-D attention mask shape')
        if mask.dtype == torch.bool:
            allowed = mask.to(device=device)
            additive = torch.zeros(mask.shape, device=device, dtype=dtype).masked_fill(~allowed, float('-inf'))
        elif mask.is_floating_point():
            if bool(torch.isnan(mask).any()) or bool(torch.isposinf(mask).any()):
                raise ValueError('Invalid additive attention mask values')
            additive = mask.to(device=device, dtype=dtype)
        else:
            raise ValueError('4-D attention mask must be boolean or additive floating point')
    else:
        raise ValueError('Attention mask must be 2-D or 4-D')
    query_index = torch.arange(past, past + queries, device=device)[:, None]
    key_index = torch.arange(keys, device=device)[None, :]
    return additive.masked_fill((key_index > query_index)[None, None, :, :], float('-inf'))


class SPORALayerReplacement(nn.Module):
    """A distilled SPORA causal block with learned absolute position input.

    It accepts Qwen2 decoder-layer calls and supports a mutable cache for
    prefill and single-token continuation. Position embeddings remain bounded
    by the checkpoint's configured maximum length.
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

    def extend_positions(self, max_length):
        """Preserve learned positions and zero-initialize new, untrained positions.

        Call before constructing an optimizer, because this replaces a parameter.
        Extending the table does not establish long-context model quality.
        """
        if type(max_length) is not int or max_length <= self.max_length:
            raise ValueError('Extended position limit must be a larger integer')
        old = self.position.weight
        weights = torch.zeros((max_length, self.width), device=old.device, dtype=old.dtype)
        replacement = nn.Embedding.from_pretrained(weights, freeze=not old.requires_grad)
        with torch.no_grad():
            replacement.weight.zero_()
            replacement.weight[:self.max_length].copy_(old)
        replacement.weight.requires_grad_(old.requires_grad)
        self.position = replacement
        self.max_length = max_length

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
        mask = _attention_mask(attention_mask, batch, self.block.heads, length, length, 0,
                               values.device, values.dtype)
        return self.block(values, attention_mask=mask).to(device=hidden_states.device, dtype=hidden_dtype)

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
        mask = _attention_mask(attention_mask, b, self.block.heads, n, past_length + n, past_length,
                               query.device, query.dtype)
        cache_kwargs = {} if cache_position is None else {'cache_position': cache_position}
        key, value = past_key_values.update(key, value, self.layer_idx, cache_kwargs)
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


def apply_position_tails(blocks, payload, expected_base_sha256, expected_sources):
    """Apply a checked tail-only experiment to its exact source candidates.

    Validate the entire payload before changing any block. Construct an
    optimizer only after applying, since extending replaces each position table.
    Caller must compute expected source-file digests from the loaded candidates.
    """
    if (not isinstance(blocks, dict) or not blocks or any(
            type(key) is not int or not isinstance(value, SPORALayerReplacement)
            for key, value in blocks.items())):
        raise ValueError('Expected indexed SPORA replacement blocks')
    keys = {str(key) for key in blocks}
    if (not isinstance(payload, dict) or payload.get('format') != 'spora-position-tail-v1'
            or payload.get('base_weight_sha256') != expected_base_sha256
            or not isinstance(expected_sources, dict) or set(expected_sources) != keys
            or payload.get('sources') != expected_sources):
        raise ValueError('Position tail provenance mismatch')
    old, new, tails = payload.get('old_length'), payload.get('new_length'), payload.get('tails')
    if (type(old) is not int or type(new) is not int or old <= 0 or new <= old
            or not isinstance(tails, dict) or set(tails) != keys):
        raise ValueError('Invalid position tail metadata')
    for index, block in blocks.items():
        tensor = tails[str(index)]
        if (block.layer_idx != index or block.max_length != old or not isinstance(tensor, torch.Tensor)
                or not tensor.is_floating_point() or tensor.shape != (new-old, block.width)
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError('Invalid position tail tensor or source block')
    for index, block in blocks.items():
        block.extend_positions(new)
        with torch.no_grad():
            block.position.weight[old:].copy_(tails[str(index)].to(
                device=block.position.weight.device, dtype=block.position.weight.dtype))
