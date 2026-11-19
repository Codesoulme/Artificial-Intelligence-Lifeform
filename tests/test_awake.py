import copy,tempfile,unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch
import torch
from spora.awake import AwakeConfig,AwakeCore,attach_awake
from spora.awake_chat import awake_chat_lines
from spora.awake_recall_intent import FORMAT as RECALL_INTENT_FORMAT, RecallIntent, SameSubjectIntent
from spora.awake_session import AwakeSession
from spora.awake_topics import AwakeTopics, supports_field
from spora.language_training import atomic_save
from spora.temporal import TemporalSPORA,TemporalConfig
from spora.growing import new_state,save,load,grow
from spora.state_codec import flatten
from spora.durable import TensorStore


class AwakeChatTests(unittest.TestCase):
    def test_recall_intent_checkpoint_requires_matching_decoder(self):
        source = RecallIntent(2, .7)
        with torch.no_grad():
            source.head.weight.zero_()
            source.head.bias.fill_(2.)
        fingerprint = {'model.safetensors': 'model-a'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'intent.pt'
            payload = dict(format=RECALL_INTENT_FORMAT, fingerprint=fingerprint, cutoff=.7,
                           state=source.head.state_dict())
            torch.save(payload, path)
            loaded = RecallIntent.load(path, 2, fingerprint)
            self.assertTrue(loaded(torch.ones(2), torch.ones(3, 2)))
            with self.assertRaisesRegex(ValueError, 'different format'):
                SameSubjectIntent.load(path, 2, fingerprint)
            with self.assertRaisesRegex(ValueError, 'different decoder'):
                RecallIntent.load(path, 2, {'model.safetensors': 'model-b'})
            torch.save(dict(payload, cutoff=float('nan')), path)
            with self.assertRaisesRegex(ValueError, 'cutoff'):
                RecallIntent.load(path, 2, fingerprint)

    def test_read_binding_requires_matching_activity_and_is_consumed_once(self):
        topics = AwakeTopics(('drink', 'work'))
        topics.observe([.95, .05])
        topics.advance(500_000_000)
        topics.bind_read(0, 'drink')
        subject, field, confidence = topics.consume_read()
        self.assertEqual((subject, field), (0, 'drink'))
        self.assertAlmostEqual(confidence, .95)
        self.assertIsNone(topics.consume_read())

        topics.bind_read(1, 'work')
        self.assertIsNone(topics.consume_read())
        topics.observe([.05, .95])
        topics.advance(500_000_000)
        self.assertIsNone(topics.consume_read())
        topics.bind_read(1, 'work')
        topics.invalidate()
        topics.observe([.05, .95])
        topics.advance(500_000_000)
        self.assertIsNone(topics.consume_read())

    def test_read_binding_is_absent_from_checkpoint(self):
        topics = AwakeTopics(('drink', 'work'))
        topics.observe([.95, .05])
        topics.advance(500_000_000)
        topics.bind_read(0, 'drink')
        state = topics.state_dict()
        self.assertEqual(set(state), {'activity', 'drive', 'observations'})
        restored = AwakeTopics(topics.labels)
        restored.load_state_dict(state)
        self.assertIsNone(restored.consume_read())

    def test_topic_support_requires_active_and_matching_prior(self):
        self.assertTrue(supports_field({'drink': .6, 'work': .1}, 'drink'))
        self.assertFalse(supports_field({'drink': .1, 'work': .6}, 'drink'))
        self.assertFalse(supports_field({'drink': .12, 'work': .1}, 'drink'))
        self.assertFalse(supports_field({'drink': .1, 'work': .01}, 'drink'))
        self.assertFalse(supports_field(None, 'drink'))

    def test_topic_posterior_changes_recurrence_and_expires_with_elapsed_time(self):
        drink, work, no_input = (AwakeTopics(('drink', 'work')) for _ in range(3))
        drink.observe([.9, .1])
        work.observe([.1, .9])
        for trace in (drink, work, no_input):
            trace.advance(500_000_000)
        self.assertEqual(int(drink.activity.argmax()), 0)
        self.assertEqual(int(work.activity.argmax()), 1)
        torch.testing.assert_close(drink.activity, work.activity.flip(0))
        self.assertEqual(float(no_input.activity.sum()), 0.)
        initial = float(drink.activity.sum())
        for _ in range(40):
            drink.advance(250_000_000)
        self.assertLess(float(drink.activity.sum()), initial / 10)
        drink.invalidate()
        self.assertEqual(float(drink.activity.sum()), 0.)
        self.assertEqual(int(drink.observations), 1)

    def test_clock_and_periodic_save_continue_while_chat_processes_a_line(self):
        model = torch.nn.Linear(1, 1)
        core = AwakeCore(AwakeConfig(relax_seconds=.03, focus_seconds=.01))
        checkpointed = Event()

        def observe_save(payload, path):
            atomic_save(payload, path)
            checkpointed.set()

        with tempfile.TemporaryDirectory() as directory:
            trained, saved = (Path(directory) / name for name in ('trained.pt', 'awake.pt'))
            torch.save(dict(config=core.config_dict(), state=core.state_dict()), trained)
            with (patch('spora.awake_chat.SAVE_INTERVAL_NS', 100_000_000),
                  patch('spora.awake_chat.atomic_save', side_effect=observe_save),
                  awake_chat_lines(model, ['question'], saved, trained, {}, lambda status: None,
                                   topic_labels=('drink', 'work')) as (lines, observe, _, _bind)):
                self.assertEqual(next(lines), 'question')
                observe([.1, .9])
                self.assertTrue(checkpointed.wait(timeout=5))
                state = torch.load(saved, weights_only=True)['core']['state']
                self.assertGreater(int(state['neural_steps']), 0)
                self.assertEqual(int(state['gap_ns']), 0)
                self.assertEqual(list(lines), [])
            final = torch.load(saved, weights_only=True)['core']['state']
            self.assertGreaterEqual(int(final['neural_steps']), int(state['neural_steps']))
            topics = torch.load(saved, weights_only=True)['topics']['state']
            self.assertEqual(int(topics['observations']), 1)
            self.assertGreater(float(topics['activity'][1]), float(topics['activity'][0]))


class AwakeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        self.model=TemporalSPORA(TemporalConfig(width=32,layers=2,heads=2,
            memory_width=64,expert_rank=4,memory_layers=(0,1),history_writer='rls',current_address='orthogonal'))
        self.state=new_state(self.model);attach_awake(self.model,AwakeCore())
        self.core=self.model.awake_core

    def test_sleep_stops_computation_and_wake_requires_new_observation(self):
        self.core.observe(torch.eye(16),0,'synthetic');self.core.set_goal(3)
        self.core.tick(100);before=self.core.values.clone();count=int(self.core.neural_steps)
        self.core.set_mode('sleep');self.core.tick(100)
        self.assertEqual(int(self.core.neural_steps),count)
        self.assertTrue(torch.equal(before,self.core.values))
        self.assertEqual(int(self.core.sleep_ns),100)
        self.core.set_mode('relax');self.assertFalse(bool(self.core.observation_valid))
        self.assertIsNone(self.core.status()['proposed_node'])

    def test_no_input_runs_neural_update_without_fabricating_observation(self):
        calls=[]
        hook=self.core.transition.register_forward_hook(lambda *args:calls.append(True))
        self.core.tick(100);hook.remove()
        self.assertGreater(int(self.core.neural_steps),0)
        self.assertEqual(int(self.core.observations),0)
        self.assertEqual(int(self.core.awake_ns),100)
        self.assertTrue(calls)  # A learned fixed point may legitimately stay unchanged.

    def test_gap_and_clock_reversal_never_create_missing_thoughts(self):
        mono=[100];utc=[1000]
        session=AwakeSession(self.model,lambda:mono[0],lambda:utc[0])
        self.core.observe(torch.eye(16),0,'sensor');self.core.set_goal(3)
        mono[0]+=3_000_000_000;session.tick()
        self.assertEqual(int(self.core.neural_steps),0)
        self.assertEqual(int(self.core.awake_ns),0)
        self.assertEqual(int(self.core.gap_ns),3_000_000_000)
        mono[0]-=1
        with self.assertRaises(ValueError):session.tick()

    def test_restart_invalidates_sensors_even_without_prior_timestamp(self):
        self.core.observe(torch.eye(16),0,'sensor');self.core.set_goal(3)
        session=AwakeSession(self.model,lambda:100,lambda:1000)
        self.assertFalse(session.offline_gap_known)
        self.assertFalse(bool(self.core.observation_valid))
        self.assertEqual(int(self.core.target),3)
        self.assertEqual(int(self.core.observations),1)

    def test_elapsed_duration_is_not_number_of_calls(self):
        other=copy.deepcopy(self.core)
        for _ in range(10):self.core.tick(100_000_000)
        for _ in range(2):other.tick(500_000_000)
        self.assertEqual(int(self.core.awake_ns),int(other.awake_ns))
        self.assertNotEqual(int(self.core.ticks),int(other.ticks))

    def test_new_observation_invalidates_prior_proposal_and_rejects_bad_input(self):
        self.core.observe(torch.eye(16),0,'sensor');self.core.set_goal(3)
        self.core.tick(100);revision=int(self.core.revision)
        self.core.observe(torch.zeros(16,16),1,'user',urgent=True)
        self.assertEqual(int(self.core.revision),revision+1)
        self.assertEqual(int(self.core.iterations),0)
        self.assertFalse(bool(self.core.readiness))
        before=copy.deepcopy(self.core.state_dict())
        with self.assertRaises(ValueError):self.core.observe(torch.full((16,16),float('nan')),0,'sensor')
        self.assertTrue(all(torch.equal(v,self.core.state_dict()[k]) for k,v in before.items()))

    def test_portable_transaction_and_growth_preserve_internal_work_state(self):
        self.core.observe(torch.eye(16),0,'synthetic');self.core.set_goal(3);self.core.tick(100)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'awake.pt';save(path,self.model,self.state)
            restored,state,_=load(path)
            self.assertTrue(all(torch.equal(v,restored.awake_core.state_dict()[k]) for k,v in self.core.state_dict().items()))
            self.core.tick(100);restored.awake_core.tick(100)
            self.assertTrue(torch.equal(self.core.values,restored.awake_core.values))
            with TensorStore.create(Path(directory)/'awake.sqlite',self.model,self.state) as store:
                self.core.tick(100)
                def fail(stage):
                    if stage=='after_tensors':raise RuntimeError('injected crash')
                with self.assertRaises(RuntimeError):store.commit(self.model,self.state,store.revision,hook=fail)
                old,_,_,_=store.read()
                self.assertEqual(int(old.awake_core.ticks),2)
                store.commit(self.model,self.state,store.revision)
                new,_,_,_=store.read();self.assertEqual(int(new.awake_core.ticks),3)
            grown=grow(self.model,self.state)
            schema,_=flatten(self.model,grown);self.assertIn('awake_core',schema)


if __name__=='__main__':unittest.main()
