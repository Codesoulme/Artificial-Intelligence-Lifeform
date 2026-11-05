import tempfile
from pathlib import Path
import unittest
import torch
from spora.temporal import TemporalSPORA,TemporalConfig,reset_bank
from spora.temporal_tasks import events,queries
from spora.checkpoint import save_checkpoint,load_checkpoint
from spora.temporal_cli import create,learn
from spora.temporal_infer import validate_request
from spora.p4 import task_loss


class TemporalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(84)
        self.model=TemporalSPORA(TemporalConfig(width=32,layers=2,heads=2,memory_width=16,
                                               expert_rank=4,memory_layers=(0,1)))

    def test_environment_separates_observed_from_true(self):
        r=events(torch.Generator().manual_seed(1),2,8,self.model.config,'cpu',offset=0)
        q=queries(r,7,self.model.config)
        self.assertTrue(torch.equal(q['current'][3][:,0],r['values'][:,4]))
        self.assertTrue(torch.equal(q['history'][3][:,5],r['values'][:,5]))
        self.assertTrue(bool((q['invalid'][3]==self.model.config.unknown).all()))

    def test_untrusted_event_only_changes_history(self):
        f,v=self.model.initial_state(1)
        updated,_=self.model.write_temporal(f,v,torch.tensor([3]),torch.tensor([5]),torch.tensor([2]),torch.zeros(1))
        for old,new in zip(f,updated):
            torch.testing.assert_close(old[0],new[0],atol=0,rtol=0)
            self.assertFalse(torch.equal(old[1],new[1]))
        key=torch.tensor([3])
        torch.testing.assert_close(self.model(key,f),self.model(key,updated),atol=0,rtol=0)

    def test_current_query_cannot_depend_on_time(self):
        f,_=self.model.initial_state(1);key=torch.tensor([3]);mode=torch.zeros_like(key)
        torch.testing.assert_close(self.model(key,f,mode,torch.tensor([1])),
                                   self.model(key,f,mode,torch.tensor([63])),atol=0,rtol=0)

    def test_temporal_gradient_and_checkpoint(self):
        f,v=self.model.initial_state(1,True)
        f,v=self.model.write_temporal(f,v,torch.tensor([3]),torch.tensor([5]),torch.tensor([2]),torch.ones(1),True)
        self.model(torch.tensor([3]),f,torch.ones(1,dtype=torch.long),torch.tensor([2])).sum().backward()
        self.assertIsNotNone(self.model.history_binding.weight.grad)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'time.pt';save_checkpoint(p,self.model,f,v,cursor=3)
            model,ff,_,_=load_checkpoint(p)
            self.assertIsInstance(model,TemporalSPORA)
            torch.testing.assert_close(model(torch.tensor([3]),ff),self.model(torch.tensor([3]),f),atol=0,rtol=0)

    def test_invalid_time_and_unknown_write_rejected(self):
        f,v=self.model.initial_state(1)
        for value,time in [(17,1),(16,1),(3,64),(3,-1)]:
            with self.assertRaises(ValueError):
                self.model.write_temporal(f,v,torch.tensor([3]),torch.tensor([value]),torch.tensor([time]),torch.ones(1))

    def test_history_reset_does_not_change_current(self):
        f,v=self.model.initial_state(1)
        f,v=self.model.write_temporal(f,v,torch.tensor([3]),torch.tensor([5]),torch.tensor([2]),torch.ones(1))
        reset=reset_bank(self.model,f,1)
        torch.testing.assert_close(self.model(torch.tensor([3]),f),self.model(torch.tensor([3]),reset),atol=0,rtol=0)

    def test_cli_atomic_continuation_and_capacity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);foundation=root/'base.pt';individual=root/'person.pt'
            save_checkpoint(foundation,self.model)
            create(individual,foundation)
            original=individual.read_bytes()
            with self.assertRaises(FileExistsError):create(individual,foundation)
            self.assertEqual(original,individual.read_bytes())
            learn(individual,3,5,1)
            _,first,_,data=load_checkpoint(individual)
            self.assertEqual(data['cursor'],1)
            learn(individual,3,7,0)
            model,second,velocity,data=load_checkpoint(individual)
            self.assertEqual(data['cursor'],2)
            for a,b in zip(first,second):torch.testing.assert_close(a[0],b[0],atol=0,rtol=0)
            self.assertFalse(torch.equal(first[0][1],second[0][1]))
            saved=individual.read_bytes()
            with self.assertRaises(ValueError):learn(individual,3,model.config.unknown,1)
            self.assertEqual(saved,individual.read_bytes())
            save_checkpoint(individual,model,second,velocity,cursor=64)
            saved=individual.read_bytes()
            with self.assertRaises(ValueError):learn(individual,3,9,1)
            self.assertEqual(saved,individual.read_bytes())
            self.assertFalse(individual.with_suffix('.lock').exists())

    def test_existing_lock_is_not_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'person.pt';lock=path.with_suffix('.lock');lock.write_text('writer')
            with self.assertRaises(FileExistsError):learn(path,3,4,1)
            self.assertEqual(lock.read_text(),'writer')

    def test_inference_refuses_history_and_coercion(self):
        for bad in ({'keys':[1],'modes':[0],'times':[0],'answers':[5]},
                    {'keys':[1.5],'modes':[0],'times':[0]},
                    {'keys':[True],'modes':[0],'times':[0]},
                    {'keys':[1],'modes':[0,1],'times':[0]},
                    {'keys':[],'modes':[],'times':[]}):
            with self.assertRaises(ValueError):validate_request(bad)

    def test_last_time_is_not_supervised_as_future(self):
        record=events(torch.Generator().manual_seed(9),1,8,self.model.config,'cpu',offset=56)
        groups=queries(record,8,self.model.config)
        self.assertFalse(bool(groups['future_mask'].any()))
        fast,_=self.model.initial_state(1)
        original=task_loss(self.model,fast,groups)
        k,m,t,y=groups['future'];groups['future']=(k,m,t,torch.zeros_like(y))
        torch.testing.assert_close(original,task_loss(self.model,fast,groups),atol=0,rtol=0)

    def test_legacy_writer_rejects_temporal_model(self):
        fast,velocity=self.model.initial_state(1)
        with self.assertRaises(ValueError):
            self.model.write_event(fast,velocity,torch.tensor([3]),torch.tensor([5]))

    def test_invalid_cursor_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bad.pt';save_checkpoint(path,self.model,cursor=65)
            with self.assertRaises(ValueError):load_checkpoint(path)

    def test_shared_untrusted_updates_are_not_claimed_as_noop(self):
        config=TemporalConfig(width=32,layers=2,heads=2,memory_width=16,
                              expert_rank=4,memory_layers=(0,1),temporal_kind='shared')
        model=TemporalSPORA(config);fast,velocity=model.initial_state(1)
        updated,_=model.write_temporal(fast,velocity,torch.tensor([3]),torch.tensor([5]),torch.tensor([2]),torch.zeros(1))
        self.assertFalse(torch.equal(fast[0][0],updated[0][0]))
        self.assertFalse(torch.equal(model(torch.tensor([3]),fast),model(torch.tensor([3]),updated)))
        with self.assertRaises(ValueError):reset_bank(model,updated,1)

    def test_delta_collateral_follows_address_overlap(self):
        site=self.model.memories[0];fast,velocity=self.model.initial_state(1)
        key=torch.tensor([3]);mode=torch.ones_like(key);time=torch.tensor([2])
        _,bound=self.model.representation(key,mode,time);q=site.address(bound)
        _,other=self.model.representation(key,mode,torch.tensor([3]));r=site.address(other)
        target=site.target(self.model.embedding(torch.tensor([5])+self.model.config.value_start))
        before=site.retrieve([fast[0][1]],r)
        updated,_=site.write_temporal(fast[0],velocity[0],q,target,torch.ones(1),mode,False)
        delta=site.retrieve([updated[1]],r)-before
        expected=torch.sigmoid(site.log_lr)*(target.detach()-site.retrieve([fast[0][1]],q))*(q*r).sum(-1,keepdim=True)
        torch.testing.assert_close(delta,expected,atol=1e-6,rtol=1e-5)


if __name__=='__main__':unittest.main()
