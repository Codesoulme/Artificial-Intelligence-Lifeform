"""Finite online timing runs; never starts a hidden persistent service."""
import argparse
import json
import math
import time
import torch
from .durable import TensorStore
from .online_time import OnlineSession


def run(store, seconds, tick_ms, synthetic_pulse_period=None, demo_world=False):
    if not math.isfinite(seconds) or not .1 <= seconds <= 60 or not math.isfinite(tick_ms) or not 10 <= tick_ms <= 1000:
        raise ValueError('Run must be 0.1–60 seconds, tick 10–1000 milliseconds')
    if synthetic_pulse_period is not None and (not math.isfinite(synthetic_pulse_period) or not .2 <= synthetic_pulse_period <= 2):
        raise ValueError('Synthetic pulse period must be 0.2–2 seconds')
    model, state, _, revision = store.read()
    if state.runtime is None:
        raise ValueError('Individual has no time module')
    session = OnlineSession(model.time_world, state.runtime)
    if demo_world:
        session.observe_world([0., .4, -.2], [.5, .2], [.4, .3], 'synthetic')
    pulses = 0
    if synthetic_pulse_period is not None:
        session.pulse('synthetic')
        pulses = 1
    started, cpu_started = time.perf_counter(), time.process_time()
    next_pulse = started+(synthetic_pulse_period or seconds+1)
    next_save = started+.5
    ticks, events, latencies, commits = 0, [], [], []
    while time.perf_counter()-started < seconds:
        time.sleep(min(tick_ms/1000, max(0, seconds-(time.perf_counter()-started))))
        now = time.perf_counter()
        if synthetic_pulse_period is not None and pulses < 3 and now >= next_pulse:
            # Emit one actual test input now. Never backfill missed pulses.
            session.pulse('synthetic')
            pulses += 1
            next_pulse = now+synthetic_pulse_period
        before = time.perf_counter()
        result = session.tick()
        latencies.append((time.perf_counter()-before)*1000)
        ticks += 1
        if result['new_expected_signal']:
            events.append({'elapsed_seconds': now-started, 'kind': 'expected_signal_due',
                           'is_observation': False})
        if now >= next_save:
            before = time.perf_counter()
            commits.append(dict(session.checkpoint(store, model, state), ms=(time.perf_counter()-before)*1000,
                                world_observations=state.runtime.world_count, observed_pulses=state.runtime.pulse_count))
            next_save = time.perf_counter()+.5
    before = time.perf_counter()
    commits.append(dict(session.checkpoint(store, model, state), ms=(time.perf_counter()-before)*1000,
                        world_observations=state.runtime.world_count, observed_pulses=state.runtime.pulse_count))
    return {'wall_seconds': time.perf_counter()-started, 'cpu_seconds': time.process_time()-cpu_started,
            'ticks': ticks, 'tick_ms_mean': sum(latencies)/max(1, len(latencies)),
            'tick_ms_max': max(latencies, default=0), 'events': events, 'commits': commits,
            'cursor': state.cursor, 'starting_revision': revision, 'final_revision': store.revision,
            'status': session.status(), 'source_note': 'Optional demo observations are synthetic; scheduling uses the real OS clock.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--individual', required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('info')
    runner = sub.add_parser('run')
    runner.add_argument('--seconds', type=float, required=True)
    runner.add_argument('--tick-ms', type=float, default=50)
    runner.add_argument('--synthetic-pulse-period', type=float)
    runner.add_argument('--demo-world', action='store_true')
    a = parser.parse_args()
    torch.set_num_threads(2)
    with TensorStore(a.individual) as store:
        if a.command == 'info':
            model, state, _, revision = store.read()
            result = {'revision': revision, 'cursor': state.cursor,
                      'runtime_checkpoint': state.runtime.metadata() if state.runtime else None,
                      'note': 'Saved snapshot only. Starting a new run requires fresh observations.'}
        else:
            result = run(store, a.seconds, a.tick_ms, a.synthetic_pulse_period, a.demo_world)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
