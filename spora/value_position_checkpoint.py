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
           or component.layer_idx != index or component.parent is not values.get(index)
           for index, component in components.items()):
        raise ValueError('Query/key positions require the same layer and value position table')
    specs = {str(index): list(component.pairs) for index, component in sorted(components.items())}
    tensors = {}
    for index, component in components.items():
        prefix = f'{QUERY_KEY_POSITION_PREFIX}{index}/'
        tensors[prefix + 'weight'] = component.weight.detach().cpu().clone()
        tensors[prefix + 'bias'] = component.bias.detach().cpu().clone()
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
    used = {f'{QUERY_KEY_POSITION_PREFIX}{name}/{field}' for name in specs for field in ('weight', 'bias')}
    if used != set(tensors):
        raise ValueError('Missing or unexpected query/key position tensors')
    decoded = {}
    for name, pairs in specs.items():
        index = values_by_name[name]
        _, _, value_weight, _ = decoded_values[index]
        width, hidden = value_weight.shape
        if (not isinstance(pairs, list) or not pairs or width % 2
                or any(type(pair) is not int or not 0 <= pair < width // 2 for pair in pairs)
                or pairs != sorted(set(pairs))):
            raise ValueError('Invalid query/key rotary pairs')
        heads = config['num_attention_heads']
        query_heads = heads // config.get('num_key_value_heads', heads)
        projection_width = 2 * len(pairs) * (query_heads + 1)
        prefix = f'{QUERY_KEY_POSITION_PREFIX}{name}/'
        weight, bias = (tensors[prefix + field] for field in ('weight', 'bias'))
        if (any(not isinstance(tensor, torch.Tensor) or tensor.dtype not in (torch.float16, torch.float32)
                or not bool(torch.isfinite(tensor).all()) for tensor in (weight, bias))
                or weight.shape != (projection_width, hidden) or bias.shape != (projection_width,)):
            raise ValueError('Invalid learned query/key position tensors')
        decoded[index] = pairs, weight, bias
    return decoded
