"""Short-lived field activations driven by the chat field router."""

import math

import torch
from torch import nn


def supports_field(activity, field):
    if not activity or field not in activity:
        return False
    mass = sum(activity.values())
    return (mass >= .2 and max(activity, key=activity.get) == field and activity[field] / mass >= .65)


class AwakeTopics(nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.labels = tuple(labels)
        self.register_buffer('drive', torch.zeros(len(self.labels)))
        self.register_buffer('activity', torch.zeros_like(self.drive))
        self.register_buffer('observations', torch.zeros((), dtype=torch.long))
        self.register_buffer('binding', torch.full((3,), -1, dtype=torch.long), persistent=False)

    def bind_read(self, entity, field, remaining=1):
        if type(entity) is not int or not 0 <= entity < 2**63:
            raise ValueError('Expected a nonnegative memory address')
        if type(remaining) is not int or not 0 < remaining < 2**63:
            raise ValueError('Expected a positive number of remaining turns')
        index = self.labels.index(field)
        self.binding[0] = entity
        self.binding[1] = index
        self.binding[2] = remaining

    def consume_read(self):
        entity, index, remaining = self.binding.tolist()
        if entity < 0:
            return None
        self.binding.fill_(-1)
        field = self.labels[index]
        activity = self.status()['activity']
        if supports_field(activity, field):
            return entity, field, activity[field] / sum(activity.values()), remaining - 1
        return None

    def observe(self, probabilities):
        values = torch.as_tensor(probabilities, dtype=self.drive.dtype, device=self.drive.device)
        if (values.shape != self.drive.shape or not bool(torch.isfinite(values).all())
                or bool((values < 0).any()) or not math.isclose(float(values.sum()), 1., abs_tol=1e-5)):
            raise ValueError('Expected normalized field probabilities in router order')
        self.drive.copy_(values)
        self.observations.add_(1)

    def advance(self, elapsed_ns):
        seconds = elapsed_ns / 1e9
        self.activity.lerp_(self.drive, -math.expm1(-seconds / .5))
        self.drive.mul_(math.exp(-seconds / 2.))

    def invalidate(self):
        self.drive.zero_()
        self.activity.zero_()
        self.binding.fill_(-1)

    def status(self):
        return dict(observations=int(self.observations),
                    activity=dict(zip(self.labels, self.activity.tolist(), strict=True)))
