"""Fresh random mappings: targets are never included in query input."""
import torch


def mappings(generator, batch_size, pairs, config, device):
    if not 1 <= pairs <= config.key_count:
        raise ValueError('Invalid pair count')
    keys = torch.stack([torch.randperm(config.key_count, generator=generator)[:pairs]
                        for _ in range(batch_size)])
    values = torch.randint(config.value_count, (batch_size, pairs), generator=generator)
    return keys.to(device), values.to(device)


def absorb(model, keys, values, fast=None, velocity=None, differentiable=False):
    if fast is None:
        fast, velocity = model.initial_state(keys.shape[0], differentiable=differentiable)
    if velocity is None:
        raise ValueError('Momentum must accompany existing fast parameters')
    for column in range(keys.shape[1]):
        fast, velocity = model.write_event(fast, velocity, keys[:, column],
                                           values[:, column], differentiable=differentiable)
    return fast, velocity
