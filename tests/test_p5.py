import tempfile,unittest
from pathlib import Path
import torch
from spora.temporal import TemporalSPORA,TemporalConfig,reset_bank
from spora.checkpoint import save_checkpoint,load_checkpoint

class RlsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(47)
        self.model=TemporalSPORA(TemporalConfig(width=32,layers=2,heads=2,memory_width=16,expert_rank=4,memory_layers=(0,1),history_writer='rls'))

    def test_rls_matches_batch_ridge_solution(self):
        site=self.model.memories[0];f,v=self.model.initial_state(1)
        initial=f[0][1].clone();keys=torch.nn.functional.normalize(torch.randn(6,16),dim=-1)
        targets=torch.randn(6,16);mode=torch.ones(1,dtype=torch.long)
        state=f[0]
        for key,target in zip(keys,targets):state,_=site.write_temporal(state,v[0],key[None],target[None],torch.ones(1),mode,False)
        inverse=torch.linalg.inv(.01*torch.eye(16)+keys.T@keys)
        expected=(targets.T@keys+.01*initial[0])@inverse
        torch.testing.assert_close(state[1][0],expected,atol=3e-4,rtol=3e-4)
        torch.testing.assert_close(state[2][0],inverse,atol=3e-3,rtol=3e-3)
        self.assertGreater(float(torch.linalg.eigvalsh(state[2][0]).min()),0)

    def test_history_reset_also_resets_optimizer_evidence(self):
        f,v=self.model.initial_state(1);init,_=self.model.initial_state(1)
        f,v=self.model.write_temporal(f,v,torch.tensor([2]),torch.tensor([5]),torch.tensor([0]),torch.ones(1))
        reset=reset_bank(self.model,f,1)
        for actual,expected in zip(reset,init):
            torch.testing.assert_close(actual[1],expected[1],atol=0,rtol=0)
            torch.testing.assert_close(actual[2],expected[2],atol=0,rtol=0)

    def test_rls_restart_and_continuation(self):
        f,v=self.model.initial_state(1)
        f,v=self.model.write_temporal(f,v,torch.tensor([2]),torch.tensor([5]),torch.tensor([0]),torch.ones(1))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'rls.pt';save_checkpoint(path,self.model,f,v,cursor=1)
            model,ff,vv,data=load_checkpoint(path)
            args=(torch.tensor([2]),torch.tensor([7]),torch.tensor([1]),torch.zeros(1))
            a,_=self.model.write_temporal(f,v,*args);b,_=model.write_temporal(ff,vv,*args)
            for aa,bb in zip(a,b):
                for x,y in zip(aa,bb):torch.testing.assert_close(x,y,atol=0,rtol=0)

    def test_orthogonal_current_does_not_interfere(self):
        model=TemporalSPORA(TemporalConfig(width=32,layers=2,heads=2,memory_width=64,expert_rank=4,memory_layers=(0,1),history_writer='rls',current_address='orthogonal'))
        fast,velocity=model.initial_state(1)
        site=model.memories[0];basis=site.current_basis
        torch.testing.assert_close(basis@basis.T,torch.eye(64),atol=1e-6,rtol=1e-5)
        old=torch.bmm(fast[0][0],basis[2][None,:,None]).squeeze(-1)
        updated,_=model.write_temporal(fast,velocity,torch.tensor([3]),torch.tensor([7]),torch.tensor([0]),torch.ones(1))
        new=torch.bmm(updated[0][0],basis[2][None,:,None]).squeeze(-1)
        torch.testing.assert_close(old,new,atol=1e-6,rtol=1e-5)

if __name__=='__main__':unittest.main()
