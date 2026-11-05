"""Bounded online clock and sensor working state, separate from neural history.

Only online monotonic deltas advance durations. UTC gaps are estimates, never
observations. A restart breaks observation continuity. No world simulator here.
"""
from dataclasses import dataclass, field, fields
import time
import torch


SHAPES = {'observed': (1, 3), 'predicted': (1, 3), 'action': (1, 2),
          'environment': (1, 2), 'intervals': (1, 2)}


@dataclass
class OnlineState:
    online_ns: int = 0
    checkpoint_utc_ns: int = 0
    gap_estimate_ns: int = 0
    gap_valid: bool = False
    world_age_ns: int = 0
    pulse_age_ns: int = 0
    world_count: int = 0
    pulse_count: int = 0
    consecutive_pulses: int = 0
    world_valid: bool = False
    deadline_emitted: bool = False
    world_source: str = 'none'
    pulse_source: str = 'none'
    tensors: dict = field(default_factory=lambda: {k: torch.zeros(s) for k, s in SHAPES.items()})

    def metadata(self):
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != 'tensors'}

    def validate(self):
        for name, value in self.metadata().items():
            if name in ('gap_valid', 'world_valid', 'deadline_emitted'):
                if type(value) is not bool:
                    raise ValueError('Boolean runtime flag required')
            elif name.endswith('_source'):
                if value not in ('none', 'synthetic', 'sensor'):
                    raise ValueError('Explicit observation source required')
            elif type(value) is not int or value < 0:
                raise ValueError('Nonnegative integer clock/count required')
        if self.consecutive_pulses > self.pulse_count or self.consecutive_pulses > 3:
            raise ValueError('Invalid pulse continuity')
        if self.world_valid and (self.world_count == 0 or self.world_source == 'none'):
            raise ValueError('World observation required')
        if set(self.tensors) != set(SHAPES):
            raise ValueError('Missing or unexpected runtime tensors')
        for name, tensor in self.tensors.items():
            if tensor.shape != SHAPES[name] or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
                raise ValueError('Invalid runtime tensor')


class OnlineSession:
    def __init__(self, module, state, monotonic_ns=time.perf_counter_ns, utc_ns=time.time_ns):
        state.validate()
        self.module, self.state = module, state
        self.monotonic_ns, self.utc_ns = monotonic_ns, utc_ns
        self.anchor = monotonic_ns()
        now = utc_ns()
        state.gap_valid = state.checkpoint_utc_ns > 0 and now >= state.checkpoint_utc_ns
        state.gap_estimate_ns = now-state.checkpoint_utc_ns if state.gap_valid else 0
        # A new process cannot know intervening actions or missed signals.
        state.world_valid = False
        state.consecutive_pulses = 0
        state.deadline_emitted = False

    @torch.no_grad()
    def tick(self):
        now = self.monotonic_ns()
        if now < self.anchor:
            raise ValueError('Monotonic clock moved backwards')
        delta = now-self.anchor
        self.anchor = now
        s = self.state
        s.online_ns += delta
        s.world_age_ns += delta
        s.pulse_age_ns += delta
        horizon = s.world_age_ns/1e9
        if s.world_valid and horizon <= self.module.config.max_horizon:
            t = s.tensors
            t['predicted'] = self.module.predict(t['observed'], t['action'], t['environment'], horizon).detach()
        elif s.world_valid:
            s.world_valid = False
        status = self.status()
        event = status['signal_status'] == 'due' and not s.deadline_emitted
        if event:
            s.deadline_emitted = True
        return dict(status, new_expected_signal=event)

    @torch.no_grad()
    def observe_world(self, observed, action, environment, source):
        if source not in ('synthetic', 'sensor'):
            raise ValueError('Label the source of actual input')
        device = next(self.module.parameters()).device
        values = [torch.as_tensor(v, dtype=torch.float32, device=device).reshape(1, -1).clone()
                  for v in (observed, action, environment)]
        if [tuple(v.shape) for v in values] != [(1, 3), (1, 2), (1, 2)] or not all(bool(torch.isfinite(v).all()) for v in values):
            raise ValueError('Invalid world observations')
        x, a, e = values
        if bool((x.abs() > x.new_tensor([3., 2., 2.])).any()) or bool((a.abs() > 1).any()) or bool(((e < .15) | (e > .8)).any()):
            raise ValueError('Observation outside supported training domain')
        self.tick()
        s = self.state
        s.tensors.update(observed=x, action=a, environment=e, predicted=x.clone())
        s.world_age_ns = 0
        s.world_count += 1
        s.world_valid, s.world_source = True, source

    def pulse(self, source):
        if source not in ('synthetic', 'sensor'):
            raise ValueError('Label the source of actual input')
        self.tick()
        s = self.state
        count = s.consecutive_pulses if source == s.pulse_source else 0
        interval = s.pulse_age_ns/1e9
        if count and .196 <= interval <= 2.04:
            s.tensors['intervals'] = torch.cat([s.tensors['intervals'][:, 1:],
                                               s.tensors['intervals'].new_tensor([[interval]])], -1)
            s.consecutive_pulses = min(3, count+1)
        else:
            s.consecutive_pulses = 1
        s.pulse_age_ns = 0
        s.pulse_count += 1
        s.deadline_emitted, s.pulse_source = False, source

    @torch.no_grad()
    def status(self):
        s = self.state
        period = None
        intervals = s.tensors['intervals']
        if (s.consecutive_pulses >= 3 and bool(((intervals >= .196) & (intervals <= 2.04)).all())
                and float(intervals.max()/intervals.min()) <= 1.1):
            period = float(self.module.period(intervals)[0])
        remaining = None if period is None else period-s.pulse_age_ns/1e9
        return {'online_seconds': s.online_ns/1e9, 'gap_estimate_seconds': s.gap_estimate_ns/1e9 if s.gap_valid else None,
                'world_status': 'prediction' if s.world_valid else 'needs_observation',
                'world_prediction': s.tensors['predicted'][0].tolist() if s.world_valid else None,
                'world_age_seconds': s.world_age_ns/1e9, 'world_observations': s.world_count,
                'world_source': s.world_source, 'observed_pulses': s.pulse_count,
                'signal_status': 'unknown' if remaining is None else ('due' if remaining <= 0 else 'pending'),
                'signal_period_seconds': period, 'signal_remaining_seconds': remaining,
                'signal_source': s.pulse_source}

    def checkpoint(self, store, model, growing_state):
        if growing_state.runtime is not self.state:
            raise ValueError('Session and individual state differ')
        self.tick()
        self.state.checkpoint_utc_ns = self.utc_ns()
        return store.commit(model, growing_state, store.revision)
