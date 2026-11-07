import copy,tempfile,unittest
from pathlib import Path
import torch
from spora.awake import AwakeCore,attach_awake
from spora.awake_session import AwakeSession
from spora.temporal import TemporalSPORA,TemporalConfig
from spora.growing import new_state,save,load,grow
from spora.state_codec import flatten
from spora.durable import TensorStore


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
