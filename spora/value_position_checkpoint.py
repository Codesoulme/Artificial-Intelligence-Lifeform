"""Checkpoint metadata and tensors for learned value-position components."""

import torch


VALUE_POSITION_PREFIX = 'value_position/'
QUERY_KEY_POSITION_PREFIX = 'query_key_position/'
PARAMETERS = ('position', 'weight', 'bias')


def value_position_bundle(components, compatible, config, position_limit):
    if components is not None and not isinstance(components, dict):
        raise ValueError('Invalid value-position layers')
    if not components:
        return {}, {}
    from .pretrained_attention import ValuePositionAttention

    if any(
        type(index) is not int or not isinstance(component, ValuePositionAttention)
        or component.layer_idx != index for index, component in components.items()
    ):
        raise ValueError('Invalid value-position layers')
    specs = {str(index): component.group for index, component in sorted(components.items())}
    tensors = {}
    for index, component in components.items():
        values = component.position.weight, component.weight, component.bias
        for name, tensor in zip(PARAMETERS, values, strict=True):
            tensors[f'{VALUE_POSITION_PREFIX}{index}/{name}'] = tensor.detach().cpu().clone()
    decode_value_positions(specs, tensors, compatible, config, position_limit)
    return specs, tensors


def decode_value_positions(specs, tensors, compatible, config, position_limit):
    if not isinstance(specs, dict):
        raise ValueError('Invalid value-position metadata')
    if not specs and not tensors:
        return {}
    heads = config.get('num_attention_heads')
    key_heads = config.get('num_key_value_heads', heads)
    hidden = config.get('hidden_size')
    if (type(heads) is not int or heads < 1 or type(hidden) is not int or hidden % heads
            or type(key_heads) is not int or key_heads < 1 or heads % key_heads
            or type(position_limit) is not int or position_limit < 1):
        raise ValueError('Value positions need a compatible grouped model with a fixed position limit')
    width = config.get('head_dim', hidden // heads)
    if type(width) is not int or width < 1:
        raise ValueError('Invalid value-position head width')
    by_layer = {str(index): index for index in compatible}
    if any(name not in by_layer or type(group) is not int or not 0 <= group < key_heads
           for name, group in specs.items()):
        raise ValueError('Invalid value-position layer or group')
    used = {f'{VALUE_POSITION_PREFIX}{name}/{field}' for name in specs for field in PARAMETERS}
    if used != set(tensors):
        raise ValueError('Missing or unexpected value-position tensors')
    decoded = {}
    for name, group in specs.items():
        position, weight, bias = (tensors[f'{VALUE_POSITION_PREFIX}{name}/{field}'] for field in PARAMETERS)
        if any(not isinstance(tensor, torch.Tensor) or tensor.dtype not in (torch.float16, torch.float32)
               or not bool(torch.isfinite(tensor).all()) for tensor in (position, weight, bias)):
            raise ValueError('Invalid learned value-position tensors')
        if (position.ndim != 2 or position.shape[1] != hidden
                or not position_limit <= position.shape[0] <= 65536
                or weight.shape != (width, hidden) or bias.shape != (width,)):
            raise ValueError('Value-position tensors do not match the model or its position limit')
        decoded[by_layer[name]] = group, position, weight, bias
    return decoded


def query_key_position_bundle(components, values, config):
    if components is not None and not isinstance(components, dict):
        raise ValueError('Invalid query/key position layers')
    if not components:
        return {}, {}
    from .research_partial_query_key import PartialQueryKeyPositionAttention

    if any(type(index) is not int or not isinstance(component, PartialQueryKeyPositionAttention)
           or component.layer_idx != index
           for index, component in components.items()):
        raise ValueError('Invalid query/key position layers')
    specs, tensors = {}, {}
    for index, component in sorted(components.items()):
        groups = component.migrated_groups()
        value = values.get(index)
        if groups[0].parent is not value or len({item.group for item in groups}) != len(groups):
            raise ValueError('Query/key positions require unique groups sharing the value position table')
        legacy = len(groups) == 1 and groups[0].group == value.group
        specs[str(index)] = (list(groups[0].pairs) if legacy else
                             {str(item.group): list(item.pairs) for item in groups})
        for item in groups:
            prefix = f'{QUERY_KEY_POSITION_PREFIX}{index}/'
            if not legacy:
                prefix += f'{item.group}/'
            tensors[prefix + 'weight'] = item.weight.detach().cpu().clone()
            tensors[prefix + 'bias'] = item.bias.detach().cpu().clone()
    decoded_values = {index: (value.group, value.position.weight, value.weight, value.bias)
                      for index, value in values.items()}
    decode_query_key_positions(specs, tensors, decoded_values, config)
    return specs, tensors


def decode_query_key_positions(specs, tensors, decoded_values, config):
    if not isinstance(specs, dict):
        raise ValueError('Invalid query/key position metadata')
    values_by_name = {str(index): index for index in decoded_values}
    if any(name not in values_by_name for name in specs):
        raise ValueError('Query/key positions require an existing value-position layer')
    if not specs:
        if tensors:
            raise ValueError('Missing or unexpected query/key position tensors')
        return {}
    heads = config['num_attention_heads']
    key_heads = config.get('num_key_value_heads', heads)
    groups = {str(group): group for group in range(key_heads)}
    query_heads = heads // key_heads
    decoded, used = {}, set()
    for name, layout in specs.items():
        index = values_by_name[name]
        value_group, _, value_weight, _ = decoded_values[index]
        width, hidden = value_weight.shape
        legacy = isinstance(layout, list)
        by_group = {str(value_group): layout} if legacy else layout
        if not isinstance(by_group, dict) or not by_group or any(group not in groups for group in by_group):
            raise ValueError('Invalid query/key position group metadata')
        components = []
        for label, pairs in sorted(by_group.items(), key=lambda item: groups[item[0]]):
            if (not isinstance(pairs, list) or not pairs or width % 2
                    or any(type(pair) is not int or not 0 <= pair < width // 2 for pair in pairs)
                    or pairs != sorted(set(pairs))):
                raise ValueError('Invalid query/key rotary pairs')
            projection_width = 2 * len(pairs) * (query_heads + 1)
            prefix = f'{QUERY_KEY_POSITION_PREFIX}{name}/'
            if not legacy:
                prefix += f'{label}/'
            keys = prefix + 'weight', prefix + 'bias'
            if any(key not in tensors for key in keys):
                raise ValueError('Missing or unexpected query/key position tensors')
            weight, bias = (tensors[key] for key in keys)
            if (any(not isinstance(tensor, torch.Tensor) or tensor.dtype not in (torch.float16, torch.float32)
                    or not bool(torch.isfinite(tensor).all()) for tensor in (weight, bias))
                    or weight.shape != (projection_width, hidden) or bias.shape != (projection_width,)):
                raise ValueError('Invalid learned query/key position tensors')
            components.append((groups[label], pairs, weight, bias))
            used.update(keys)
        decoded[index] = components
    if used != set(tensors):
        raise ValueError('Missing or unexpected query/key position tensors')
    return decoded
