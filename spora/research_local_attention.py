"""Reuse Qwen's bounded key/value state when testing SPORA's local attention."""

from contextlib import contextmanager

import torch


@contextmanager
def local_attention(decoder, indices, window):
    """Temporarily switch selected compatible layers to a causal local window.

    Create new generation state inside this scope and keep it within the scope.
    The decoder is reserved for this experiment until the scope exits.
    """
    model = decoder.base.model
    config = model.config
    indices = tuple(indices)
    previous_types = config.layer_types
    if (type(window) is not int or window < 2 or not indices
            or any(type(index) is not int or index not in decoder.compatible_block_indices for index in indices)
            or len(set(indices)) != len(indices)
            or model.has_sliding_layers or any(mode != 'full_attention' for mode in previous_types)):
        raise ValueError('Local attention needs a fresh window on distinct compatible full-attention layers')
    previous_window = config.sliding_window
    layers = [model.layers[index] for index in indices]
    types = list(previous_types)
    for index in indices:
        types[index] = 'sliding_attention'
    config.layer_types = types
    config.sliding_window = window
    model.has_sliding_layers = True
    for layer in layers:
        layer.attention_type = 'sliding_attention'
    try:
        yield
    finally:
        for layer in layers:
            layer.attention_type = 'full_attention'
        model.has_sliding_layers = False
        config.layer_types = previous_types
        config.sliding_window = previous_window


@contextmanager
def anchor_attention(decoder, indices, window, prefix):
    """Test a causal prefix and recent-token mask with fixed absolute positions."""
    from .integrated_checkpoint import _compatible_blocks

    blocks = dict(_compatible_blocks(decoder))
    if not 0 < prefix < window or any(index not in blocks for index in indices):
        raise ValueError('Prefix attention requires compatible decoder layers and a nonempty recent window')

    def mask_positions(module, args, kwargs):
        positions = kwargs['cache_position']
        cache = kwargs['past_key_values']
        layer = None if cache is None else cache.layers[module.layer_idx]
        retained = getattr(layer, 'retained_positions', None)
        if retained is None:
            keys = torch.arange(positions[-1] + 1, device=positions.device)
        else:
            keys = torch.cat((retained, positions))
        distance = positions[:, None] - keys
        mask = (distance >= 0) & ((distance < window - prefix) | (keys < prefix))
        mask = mask[None, None]
        original = kwargs['attention_mask']
        if original is not None:
            original = original.index_select(-1, keys)
            if original.dtype == torch.bool:
                mask = mask & original
            else:
                mask = original.masked_fill(~mask, torch.finfo(original.dtype).min)
        return args, dict(kwargs, attention_mask=mask)

    handles = [blocks[index].self_attn.register_forward_pre_hook(mask_positions, with_kwargs=True)
               for index in indices]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def anchored_cache(config, indices, window, prefix):
    """Create isolated storage for append-only decoding under the anchor mask."""
    from transformers.cache_utils import DynamicCache, DynamicLayer

    if not 0 < prefix < window - 1 or any(index <= 0 or index >= config.num_hidden_layers for index in indices):
        raise ValueError('Anchored storage needs a full first layer and room for recent tokens')

    class AnchoredLayer(DynamicLayer):
        retained_positions = None
        seen = 0

        def update(self, key_states, value_states, cache_kwargs=None):
            positions = cache_kwargs['cache_position']
            all_positions = (positions if self.retained_positions is None else
                             torch.cat((self.retained_positions, positions)))
            keys, values = super().update(key_states, value_states, cache_kwargs)
            self.seen += positions.numel()
            recent = window - prefix - 1
            if all_positions.numel() > prefix + recent:
                keep = (all_positions < prefix) | (all_positions >= self.seen - recent)
                selected = keep.nonzero(as_tuple=True)[0]
                self.keys = keys.index_select(-2, selected)
                self.values = values.index_select(-2, selected)
                self.retained_positions = all_positions.index_select(0, selected)
            else:
                self.retained_positions = all_positions
            return keys, values

        def get_seq_length(self):
            return self.seen

        def reset(self):
            super().reset()
            self.keys = self.keys[..., :0, :] if self.is_initialized else None
            self.values = self.values[..., :0, :] if self.is_initialized else None
            self.seen = 0
            self.retained_positions = None

        def crop(self, max_length):
            raise ValueError('Anchored storage supports append-only generation')

    cache = DynamicCache(config=config)
    for index in indices:
        cache.layers[index] = AnchoredLayer()
    return cache
