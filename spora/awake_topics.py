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

    def status(self):
        return dict(observations=int(self.observations),
                    activity=dict(zip(self.labels, self.activity.tolist(), strict=True)))
