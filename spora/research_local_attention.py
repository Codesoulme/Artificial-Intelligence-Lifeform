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


def _validate_landmarks(window, prefix, landmarks):
    if (not isinstance(landmarks, tuple) or any(type(i) is not int or i < prefix for i in landmarks)
            or landmarks != tuple(sorted(set(landmarks))) or prefix + len(landmarks) >= window - 1):
        raise ValueError('Landmarks must be distinct ordered positions within the anchor budget')


def _keep_key(query, keys, window, prefix, landmarks):
    distance = query - keys
    if not landmarks:
        return (distance >= 0) & ((keys < prefix) | (distance < window - prefix))
    anchors = keys.new_tensor(landmarks)
    count = torch.searchsorted(anchors, keys, right=True)
    pinned = (keys < prefix) | ((count > 0) & (anchors[(count - 1).clamp_min(0)] == keys))
    # For a non-landmark key k, only landmarks at or before k displace it.
    # The predicate is monotonic in q, so an evicted key can never be needed later.
    return (distance >= 0) & (pinned | (distance + prefix + count < window))


@torch.no_grad()
def _attended_landmarks(module, kwargs, window, prefix, count, by_group, *, required=()):
    """Rank omitted positions using normalized attention from the prompt's last 16 queries."""
    hidden = kwargs['hidden_states'].to(module.qkv.weight.dtype)
    if hidden.shape[0] != 1:
        raise ValueError('Snapshot anchors currently support one prompt per generation')
    queries, keys, _ = module.project(hidden, kwargs['position_embeddings'], kwargs['position_ids'])
    length = keys.shape[-2]
    observations = min(16, length)
    groups = queries.shape[1] // keys.shape[1]
    queries = queries[:, :, -observations:].unflatten(1, (keys.shape[1], groups)).float()
    scores = torch.einsum('bhgqd,bhkd->bhgqk', queries, keys.float()) * module.scaling
    positions = kwargs['cache_position']
    scores.masked_fill_(positions[None, :] > positions[-observations:, None], -torch.inf)
    attention = scores.softmax(-1).mean(dim=(0, 2, 3) if by_group else (0, 1, 2, 3))
    clustered = torch.nn.functional.avg_pool1d(attention.reshape(1, -1, length), 5, stride=1, padding=2)[0]
    stop = length - (window - prefix - count - 1)
    eligible = clustered[:, prefix:stop]
    if len(required) > count or any(not prefix <= position < stop for position in required):
        raise ValueError('Required positions must fit in the older-token budget')
    pinned = torch.tensor(required, dtype=torch.long, device=keys.device) - prefix
    shared = (count - len(required)) // 2 if by_group else count - len(required)
    ranked = eligible.mean(dim=0).scatter(0, pinned, -torch.inf)
    chosen = torch.cat((pinned, ranked.topk(shared).indices))
    quota, extra = divmod(count - chosen.numel(), len(eligible))
    for index, group in enumerate(eligible):
        take = quota + (index < extra)
        if take:
            remaining = group.scatter(0, chosen, -torch.inf)
            chosen = torch.cat((chosen, remaining.topk(take).indices))
    indices = chosen + prefix
    return tuple(sorted(indices.tolist()))


def install_anchor_mask(decoder, indices, window, prefix, *, landmarks=(), snapshot=0, full_prefill=False,
                        snapshot_by_group=False):
    """Install causal prefix and recent-token masks on selected decoder layers."""
    from .integrated_checkpoint import _compatible_blocks

    blocks = dict(_compatible_blocks(decoder))
    if not 0 < prefix < window or any(index not in blocks for index in indices):
        raise ValueError('Prefix attention requires compatible decoder layers and a nonempty recent window')
    _validate_landmarks(window, prefix, landmarks)
    if type(snapshot) is not int or not 0 <= snapshot < window - prefix - 1 or (landmarks and snapshot):
        raise ValueError('Snapshot anchors require a separate observed-token budget')
    if snapshot_by_group and not snapshot:
        raise ValueError('Group quotas require an observed-token budget')

    def mask_positions(module, args, kwargs):
        positions = kwargs['cache_position']
        cache = kwargs['past_key_values']
        layer = None if cache is None else cache.layers[module.layer_idx]
        past = 0 if layer is None else layer.get_seq_length()
        if past + positions.numel() <= window:
            return None
        if (snapshot or full_prefill) and layer is None:
            return None
        if (snapshot or full_prefill) and past == 0:
            if snapshot:
                if not hasattr(layer, 'landmarks'):
                    raise ValueError('Snapshot attention needs an anchored cache')
                layer.landmarks = _attended_landmarks(module, kwargs, window, prefix, snapshot, snapshot_by_group)
            return None
        retained = getattr(layer, 'retained_positions', None)
        if retained is None:
            keys = torch.arange(positions[-1] + 1, device=positions.device)
        else:
            keys = torch.cat((retained, positions))
        pinned = getattr(layer, 'landmarks', landmarks)
        mask = _keep_key(positions[:, None], keys, window, prefix, pinned)[None, None]
        original = kwargs['attention_mask']
        if original is not None:
            original = original.index_select(-1, keys)
            if original.dtype == torch.bool:
                mask = mask & original
            else:
                mask = original.masked_fill(~mask, torch.finfo(original.dtype).min)
        return args, dict(kwargs, attention_mask=mask)

    return [blocks[index].self_attn.register_forward_pre_hook(mask_positions, with_kwargs=True)
            for index in indices]


@contextmanager
def anchor_attention(decoder, indices, window, prefix, *, landmarks=(), snapshot=0, full_prefill=False,
                     snapshot_by_group=False):
    """Apply the anchor mask for an isolated experiment."""
    handles = install_anchor_mask(decoder, indices, window, prefix, landmarks=landmarks, snapshot=snapshot,
                                  full_prefill=full_prefill, snapshot_by_group=snapshot_by_group)
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def validate_anchor_layout(layout, compatible, position_limit):
    if layout is None:
        return None
    required = {'layers', 'window', 'prefix'}
    if not isinstance(layout, dict) or not required <= layout.keys() <= required | {'full_prefill', 'snapshot'}:
        raise ValueError('Invalid anchored attention layout')
    layers, window, prefix = (layout[key] for key in ('layers', 'window', 'prefix'))
    if (type(window) is not int or type(prefix) is not int or type(position_limit) is not int
            or not 0 < prefix < window - 1 or window >= position_limit
            or not isinstance(layers, (tuple, list)) or not layers
            or any(type(index) is not int or index <= 0 or index not in compatible for index in layers)
            or len(set(layers)) != len(layers)):
        raise ValueError('Anchored attention needs compatible layers and a valid context window')
    snapshot = layout.get('snapshot', 0)
    if type(layout.get('full_prefill', False)) is not bool or type(snapshot) is not int or not 0 <= snapshot < window - prefix - 1:
        raise ValueError('Invalid anchored prefill options')
    return dict(layout, layers=sorted(layers))


def anchored_cache(config, indices, window, prefix, *, landmarks=()):
    """Create isolated storage for append-only decoding under the anchor mask."""
    from transformers.cache_utils import DynamicCache, DynamicLayer

    if not 0 < prefix < window - 1 or any(index <= 0 or index >= config.num_hidden_layers for index in indices):
        raise ValueError('Anchored storage needs a full first layer and room for recent tokens')
    _validate_landmarks(window, prefix, landmarks)

    class AnchoredLayer(DynamicLayer):
        retained_positions = None
        seen = 0

        def __init__(self):
            super().__init__()
            self.landmarks = landmarks

        def update(self, key_states, value_states, cache_kwargs=None):
            positions = cache_kwargs['cache_position']
            all_positions = (positions if self.retained_positions is None else
                             torch.cat((self.retained_positions, positions)))
            keys, values = super().update(key_states, value_states, cache_kwargs)
            self.seen += positions.numel()
            if all_positions.numel() >= window:
                keep = _keep_key(self.seen, all_positions, window, prefix, self.landmarks)
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
            self.landmarks = landmarks

        def crop(self, max_length):
            raise ValueError('Anchored storage supports append-only generation')

    cache = DynamicCache(config=config)
    for index in indices:
        cache.layers[index] = AnchoredLayer()
    return cache
