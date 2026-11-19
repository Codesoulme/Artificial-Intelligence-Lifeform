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

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data['format'] != 'spora-time-world-v1':
            raise ValueError('Unknown time module')
        model = cls(WorldTimeConfig(**data['config']))
        model.load_state_dict(data['state'], strict=True)
        if any(not bool(torch.isfinite(tensor).all()) for tensor in model.parameters()):
            raise ValueError('World model weights must be finite')
        return model.eval().requires_grad_(False), data

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

    @torch.inference_mode()
    def imagine(self, state, actions, environment, durations):
        """Roll out bounded alternative actions from the same observed state."""
        if (state.shape != (1, 3) or actions.ndim != 3 or actions.shape[-1] != 2
                or not 1 <= actions.shape[0] <= 64 or not 1 <= actions.shape[1] <= 16
                or environment.shape != (1, 2) or durations.shape != (actions.shape[1],)):
            raise ValueError('Expected one observed start and 1–64 plans of at most 16 actions')
        if not all(tensor.device == state.device and tensor.dtype == state.dtype and bool(torch.isfinite(tensor).all())
                   for tensor in (state, actions, environment, durations)):
            raise ValueError('Imagined inputs must be finite with the same device and dtype')
        if (bool((state.abs() > state.new_tensor([3., 2., 2.])).any()) or bool((actions.abs() > 1.).any())
                or bool(((environment < .15) | (environment > .8)).any())):
            raise ValueError('Imagined inputs are outside the training domain')
        if bool((durations <= 0).any()) or float(durations.double().sum()) > self.config.max_horizon:
            raise ValueError('Imagined duration exceeds the validated window')
        current = state.expand(actions.shape[0], -1)
        parameters = environment.expand(actions.shape[0], -1)
        active = torch.ones(actions.shape[0], device=state.device, dtype=torch.bool)
        bound = state.new_tensor([3., 2., 2.])
        states, supported = [], []
        for action, duration in zip(actions.unbind(1), durations.unbind(), strict=True):
            predicted = self.predict(current, action, parameters, duration)
            active = active & (predicted.abs() <= bound).all(dim=-1)
            current = torch.where(active[:, None], predicted, current)
            states.append(current)
            supported.append(active)
        return torch.stack(states, dim=1), torch.stack(supported, dim=1)

    def period(self, intervals):
        if intervals.ndim != 2 or intervals.shape[-1] != 2 or not bool(torch.isfinite(intervals).all()) or bool((intervals <= 0).any()):
            raise ValueError('Two observed intervals required')
        return F.softplus(self.period_head(intervals)).squeeze(-1)
