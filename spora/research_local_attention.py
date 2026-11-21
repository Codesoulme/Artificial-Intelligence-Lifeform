"""Reuse Qwen's bounded key/value state when testing SPORA's local attention."""

from contextlib import contextmanager


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
