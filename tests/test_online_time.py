import copy
import tempfile
import unittest
from pathlib import Path
import torch
from spora.online_time import OnlineSession, OnlineState
from spora.world_time import TimeWorld
from spora.temporal import TemporalSPORA, TemporalConfig
from spora.growing import new_state, observe, grow, save, load
from spora.state_codec import flatten, restore
from spora.durable import TensorStore
from spora.sleep_memory import propose


class Clock:
    def __init__(self, mono=0, utc=10**12):
        self.mono, self.utc = mono, utc

    def advance(self, nanoseconds):
        self.mono += nanoseconds
        self.utc += nanoseconds


class OnlineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(55)
        self.module = TimeWorld()
        self.clock = Clock()
        self.state = OnlineState()
        self.session = OnlineSession(self.module, self.state, lambda: self.clock.mono, lambda: self.clock.utc)

    def observation(self, session=None):
        (session or self.session).observe_world([0., .4, -.2], [.5, .2], [.4, .3], 'synthetic')

    def model_state(self):
        model = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=16,
            expert_rank=4, memory_layers=(0, 1), history_writer='rls'))
        model.time_world = self.module
        state = new_state(model)
        state.runtime = self.state
        return model, state

    def test_tick_partition_invariance_and_duration_difference(self):
        self.observation()
        self.clock.advance(10**9)
        expected = self.session.tick()['world_prediction']
        clock = Clock()
        session = OnlineSession(self.module, OnlineState(), lambda: clock.mono, lambda: clock.utc)
        self.observation(session)
        for _ in range(100):
            clock.advance(10**7)
            actual = session.tick()['world_prediction']
        self.assertEqual(expected, actual)
        self.assertEqual(session.state.online_ns, self.state.online_ns)
        clock.advance(10**8)
        self.assertNotEqual(actual, session.tick()['world_prediction'])

    def test_unknown_and_horizon_no_observation_fabrication(self):
        self.assertEqual(self.session.tick()['world_status'], 'needs_observation')
        self.observation()
        self.clock.advance(2_000_000_001)
        status = self.session.tick()
        self.assertIsNone(status['world_prediction'])
        self.assertEqual(self.state.world_count, 1)
        self.assertEqual(self.state.pulse_count, 0)

    def test_pulses_require_two_intervals_and_deadline_is_not_observation(self):
        for i in range(3):
            if i:
                self.clock.advance(300_000_000)
            self.session.pulse('synthetic')
            if i < 2:
                self.assertEqual(self.session.status()['signal_status'], 'unknown')
        self.assertEqual(self.session.status()['signal_status'], 'pending')
        self.clock.advance(2_000_000_000)
        self.assertTrue(self.session.tick()['new_expected_signal'])
        self.assertFalse(self.session.tick()['new_expected_signal'])
        self.assertEqual(self.state.pulse_count, 3)

    def test_irregular_signals_and_source_changes_abstain(self):
        self.session.pulse('synthetic')
        self.clock.advance(200_000_000)
        self.session.pulse('synthetic')
        self.clock.advance(1_000_000_000)
        self.session.pulse('synthetic')
        self.assertEqual(self.session.status()['signal_status'], 'unknown')
        self.session.pulse('sensor')
        self.assertEqual(self.state.consecutive_pulses, 1)
        self.clock.advance(3_000_000_000)
        self.session.pulse('sensor')
        self.assertEqual(self.state.consecutive_pulses, 1)

    def test_restart_gap_rollback_and_new_monotonic_origin(self):
        self.observation()
        self.clock.advance(500_000_000)
        self.session.tick()
        self.state.checkpoint_utc_ns = self.clock.utc
        restored = copy.deepcopy(self.state)
        new = Clock(mono=17, utc=self.clock.utc+10**10)
        session = OnlineSession(self.module, restored, lambda: new.mono, lambda: new.utc)
        self.assertEqual(restored.gap_estimate_ns, 10**10)
        self.assertEqual(restored.online_ns, 500_000_000)
        self.assertFalse(restored.world_valid)
        self.assertEqual(restored.consecutive_pulses, 0)
        new.advance(200_000_000)
        new.utc -= 10**12
        self.assertEqual(session.tick()['online_seconds'], .7)
        session = OnlineSession(self.module, restored, lambda: new.mono, lambda: new.utc)
        self.assertFalse(restored.gap_valid)
        self.assertEqual(restored.online_ns, 700_000_000)

    def test_backward_monotonic_rejected_without_state_update(self):
        self.clock.mono = -1
        with self.assertRaises(ValueError):
            self.session.tick()
        self.assertEqual(self.state.online_ns, 0)

    def test_invalid_world_input_does_not_replace_observation(self):
        self.observation()
        for observed in ([float('nan'), 0, 0], [4, 0, 0], [0, 0]):
            with self.assertRaises(ValueError):
                self.session.observe_world(observed, [0, 0], [.4, .3], 'synthetic')
        self.assertEqual(self.state.world_count, 1)

    def test_neural_interface_rejects_invalid_intervals_and_horizons(self):
        for intervals in (torch.tensor([[float('nan'), .3]]), torch.tensor([[0., .3]]), torch.ones(2)):
            with self.assertRaises(ValueError):
                self.module.period(intervals)
        for horizon in (-.1, float('nan'), 2.01):
            with self.assertRaises(ValueError):
                self.module.predict(torch.zeros(1, 3), torch.zeros(1, 2), torch.ones(1, 2)*.4, horizon)

    def test_exact_roundtrip_incremental_idle_and_old_parameters(self):
        model, state = self.model_state()
        self.observation()
        with tempfile.TemporaryDirectory() as temp:
            with TensorStore.create(Path(temp)/'state.sqlite', model, state) as store:
                before_schema, before = flatten(model, state)
                fixed = {k: v.clone() for k, v in before.items() if not k.startswith('runtime/')}
                self.clock.advance(250_000_000)
                stats = self.session.checkpoint(store, model, state)
                self.assertEqual(stats['changed_tensors'], 1)
                self.assertEqual(stats['tensor_bytes_written'], 12)
                other, restored, _, _ = store.read()
                schema, tensors = flatten(model, state)
                schema2, tensors2 = flatten(other, restored)
                self.assertEqual(schema, schema2)
                for k in tensors:
                    self.assertTrue(torch.equal(tensors[k], tensors2[k]), k)
                for k in fixed:
                    self.assertTrue(torch.equal(fixed[k], tensors[k]), k)
                self.assertEqual(state.cursor, 0)
                self.assertEqual(len(state.pages), 1)
            save(Path(temp)/'state.pt', model, state)
            other, restored, _ = load(Path(temp)/'state.pt')
            self.assertEqual(flatten(other, restored)[0], schema)

    def test_growth_write_and_sleep_keep_time_extension(self):
        model, state = self.model_state()
        state = grow(model, state)
        self.assertIs(state.runtime, self.state)
        for i in range(model.config.time_count):
            state = observe(model, state, i % model.config.key_count, i % model.config.unknown, 1)
        self.assertIs(state.runtime, self.state)
        after, _ = propose(model, state, 0, rank=4, steps=1)
        self.assertIs(after.runtime, self.state)
        schema, tensors = flatten(model, after)
        restored_model, restored = restore(schema, tensors)
        self.assertTrue(hasattr(restored_model, 'time_world'))
        self.assertEqual(restored.runtime.metadata(), state.runtime.metadata())

    def test_partial_extension_and_bad_runtime_rejected(self):
        model, state = self.model_state()
        schema, tensors = flatten(model, state)
        del schema['runtime']
        with self.assertRaises(ValueError):
            restore(schema, tensors)
        state.runtime.online_ns = -1
        with self.assertRaises(ValueError):
            flatten(model, state)


if __name__ == '__main__':
    unittest.main()
