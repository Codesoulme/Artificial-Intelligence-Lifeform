"""Small learned dynamics and interval prediction; no language or world simulator.

An explicit integrator is a structural prior. The vector field and period head
are learned from random weights. Numerical timestamps alone are not learned
knowledge or evidence of subjective experience.
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class WorldTimeConfig:
    hidden: int = 64
    integration_step: float = .1
    max_horizon: float = 2.


class TimeWorld(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or WorldTimeConfig()
        if self.config.hidden < 4 or not 0 < self.config.integration_step <= 1 or not 0 < self.config.max_horizon <= 10:
            raise ValueError('Invalid bounded dynamics configuration')
        h = self.config.hidden
        self.field = nn.Sequential(nn.Linear(7, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 3))
        self.period_head = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))

    def config_dict(self):
        return asdict(self.config)

    def derivative(self, state, action, environment):
        return self.field(torch.cat([state, action, environment], -1))

    def midpoint(self, state, action, environment, dt):
        k1 = self.derivative(state, action, environment)
        return state + dt*self.derivative(state + dt*k1*.5, action, environment)

    def predict(self, state, action, environment, elapsed):
        if state.ndim != 2 or state.shape[1] != 3 or action.shape != (len(state), 2) or environment.shape != action.shape:
            raise ValueError('Batched world state, actions and environment required')
        dt = torch.as_tensor(elapsed, dtype=state.dtype, device=state.device)
        if dt.ndim == 0:
            dt = dt.expand(len(state))
        if dt.shape != (len(state),) or not bool(torch.isfinite(dt).all()) or bool((dt < 0).any()):
            raise ValueError('Finite nonnegative horizons required')
        if bool((dt > self.config.max_horizon).any()):
            raise ValueError('Prediction horizon beyond validated range')
        if not all(bool(torch.isfinite(x).all()) for x in (state, action, environment)):
            raise ValueError('Finite observations required')
        steps = max(1, math.ceil(float(dt.max())/self.config.integration_step))
        h = (dt/steps).unsqueeze(-1)
        x = state
        for _ in range(steps):
            k1 = self.derivative(x, action, environment)
            k2 = self.derivative(x+h*k1*.5, action, environment)
            k3 = self.derivative(x+h*k2*.5, action, environment)
            k4 = self.derivative(x+h*k3, action, environment)
            x = x+h*(k1+2*k2+2*k3+k4)/6
        return x

    def period(self, intervals):
        if intervals.ndim != 2 or intervals.shape[-1] != 2 or not bool(torch.isfinite(intervals).all()) or bool((intervals <= 0).any()):
            raise ValueError('Two observed intervals required')
        return F.softplus(self.period_head(intervals)).squeeze(-1)
