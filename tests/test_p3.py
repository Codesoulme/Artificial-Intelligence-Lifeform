"""Behavioral contracts for consolidation and subsequent learning."""
import tempfile
from pathlib import Path
import unittest
import torch
from spora.model import SPORA,ModelConfig
from spora.tasks import absorb
from spora.checkpoint import save_checkpoint,load_checkpoint
from spora.consolidate_p3 import rebase_delta


class P3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(35)
        self.model=SPORA(ModelConfig(width=32,layers=2,heads=2,memory_width=16,
                                    expert_rank=4,memory_layers=(0,1),memory_kind='delta'))
        self.keys=torch.tensor([[2,8,13,19]])
        self.values=torch.tensor([[3,6,2,9]])
        self.fast,self.velocity=absorb(self.model,self.keys,self.values)

    def test_rebase_preserves_all_queries_and_parent(self):
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        child,fast,velocity=rebase_delta(self.model,self.fast,self.velocity)
        allkeys=torch.arange(self.model.config.key_count)[None,:]
        with torch.no_grad():
            torch.testing.assert_close(self.model.predict_many(allkeys,self.fast),
                                       child.predict_many(allkeys,fast),rtol=1e-5,atol=2e-6)
        self.assertTrue(all(torch.equal(v,self.model.state_dict()[k]) for k,v in before.items()))
        reset,_=child.initial_state(1)
        for left,right in zip(reset,fast):
            for a,b in zip(left,right):torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertTrue(all(torch.count_nonzero(u)==0 for site in velocity for u in site))

    def test_correction_commutes_with_transfer(self):
        child,fast,velocity=rebase_delta(self.model,self.fast,self.velocity)
        changed=(self.values[:,0]+1)%self.model.config.value_count
        old_next,_=self.model.write_event(self.fast,self.velocity,self.keys[:,0],changed)
        new_next,_=child.write_event(fast,velocity,self.keys[:,0],changed)
        allkeys=torch.arange(self.model.config.key_count)[None,:]
        with torch.no_grad():
            torch.testing.assert_close(self.model.predict_many(allkeys,old_next),
                                       child.predict_many(allkeys,new_next),rtol=1e-5,atol=3e-6)

    def test_repeated_rebase_has_fixed_size_and_restart(self):
        child,fast,velocity=rebase_delta(self.model,self.fast,self.velocity)
        count=child.parameter_count()
        for _ in range(3):
            child,fast,velocity=rebase_delta(child,fast,velocity)
            self.assertEqual(child.parameter_count(),count)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'candidate.pt';save_checkpoint(path,child,fast,velocity,cursor=4)
            restored,rf,_,_=load_checkpoint(path,reset_fast=True)
            with torch.no_grad():
                torch.testing.assert_close(child(self.keys[:,0],fast),restored(self.keys[:,0],rf),atol=0,rtol=0)

    def test_rebase_rejects_multi_individual_batch(self):
        fast,velocity=self.model.initial_state(2)
        with self.assertRaises(ValueError):rebase_delta(self.model,fast,velocity)

    def test_rebase_rejects_nonfinite_without_mutation(self):
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        self.fast[0][0].data[0,0,0]=float('nan')
        with self.assertRaises(ValueError):rebase_delta(self.model,self.fast,self.velocity)
        self.assertTrue(all(torch.equal(v,self.model.state_dict()[k]) for k,v in before.items()))

    def test_residual_write_targets_total_signal_at_each_site(self):
        self.model.config.residual_write=True
        with torch.no_grad():
            for site in self.model.memories:
                site.log_lr.fill_(50)
                for expert in site.experts:
                    expert[-1].bias.add_(3)
        key=torch.tensor([8]);value=torch.tensor([11])
        fast,_=self.model.write_event(self.fast,self.velocity,key,value)
        tokens=self.model.query_tokens(key)
        with torch.no_grad():
            x=self.model.embedding(tokens)+self.model.position(torch.arange(tokens.shape[1]))
            key_embedding=self.model.embedding(key+self.model.config.key_start)
            value_embedding=self.model.embedding(value+self.model.config.value_start)
            site_index=0
            for index,block in enumerate(self.model.blocks):
                x=block(x)
                if index in self.model.config.memory_layers:
                    site=self.model.memories[site_index];query=site.address(key_embedding)
                    actual=site.retrieve(fast[site_index],query)+site.slow_signal(x,query)
                    torch.testing.assert_close(actual,site.target(value_embedding),atol=3e-6,rtol=1e-5)
                    x=site.read(x,query,fast[site_index]);site_index+=1

    def test_residual_untrusted_write_and_checkpoint_roundtrip(self):
        self.model.config.residual_write=True
        fast,velocity=self.model.write_event(self.fast,self.velocity,self.keys[:,0],self.values[:,0],trusted=torch.zeros(1))
        for old,new in zip(self.fast+self.velocity,fast+velocity):
            for a,b in zip(old,new):torch.testing.assert_close(a,b,atol=0,rtol=0)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'residual.pt';save_checkpoint(path,self.model,fast,velocity)
            restored,rf,rv,_=load_checkpoint(path)
            self.assertTrue(restored.config.residual_write)
            expected,_=self.model.write_event(fast,velocity,self.keys[:,0],torch.tensor([10]))
            resumed,_=restored.write_event(rf,rv,self.keys[:,0],torch.tensor([10]))
            for old,new in zip(expected,resumed):
                for a,b in zip(old,new):torch.testing.assert_close(a,b,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
