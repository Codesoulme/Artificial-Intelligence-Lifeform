import tempfile
from pathlib import Path
import unittest
import torch
from spora.model import SPORA,ModelConfig,DeltaMemorySite
from spora.p2 import Event,write,stream,world_loss
from spora.checkpoint import save_checkpoint,load_checkpoint


class P2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91);torch.set_num_threads(2)
        self.config=ModelConfig(width=32,layers=2,heads=2,memory_width=16,expert_rank=4,memory_layers=(0,1))
        self.model=SPORA(self.config)

    def test_delta_exact_association_and_orthogonal_invariance(self):
        site=DeltaMemorySite(self.config)
        key=torch.zeros(1,16);key[0,0]=1
        other=torch.zeros(1,16);other[0,1]=1
        params=[torch.randn(1,16,16)];velocity=[torch.zeros_like(params[0])]
        target=torch.randn(1,16)
        with torch.no_grad():site.log_lr.fill_(50)
        before=site.retrieve(params,other)
        updated,_=site.write(params,velocity,key,target,torch.ones(1),False)
        torch.testing.assert_close(site.retrieve(updated,key),target,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(site.retrieve(updated,other),before,atol=0,rtol=0)

    def test_multistep_never_writes_untrusted_rows(self):
        fast,velocity=self.model.initial_state(2)
        event=Event(torch.tensor([1,2]),torch.tensor([3,4]),torch.tensor([1.,0.]),'mixed')
        updated,_,steps=write(self.model,fast,velocity,event,'adaptive-max4',.99)
        self.assertEqual(steps[1].item(),0)
        self.assertLessEqual(steps.max().item(),4)
        for old,new in zip(fast,updated):
            for a,b in zip(old,new):torch.testing.assert_close(a[1],b[1],atol=0,rtol=0)

    def test_curriculum_truth_is_last_trusted_value(self):
        generator=torch.Generator().manual_seed(88)
        events,keys,original,current=stream(generator,2,8,self.config,'cpu')
        truth={}
        for event in events:
            if event.trusted[0]:truth[int(event.key[0])]=int(event.value[0])
        self.assertEqual([truth[int(k)] for k in keys[0]],current[0].tolist())
        self.assertTrue(bool((current[:,:4]!=original[:,:4]).all()))
        torch.testing.assert_close(current[:,4:],original[:,4:])

    def test_segment_gradients_and_fixed_backbone(self):
        events,keys,_,_=stream(torch.Generator().manual_seed(8),2,16,self.config,'cpu')
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        fast,velocity=self.model.initial_state(2,True)
        loss,updates=world_loss(self.model,events,keys,fast,velocity,'two',.8)
        self.assertGreater(updates,0);self.assertGreater(loss,0)
        self.assertTrue(all(torch.equal(v,self.model.state_dict()[k]) for k,v in before.items()))
        for site in self.model.memories:self.assertTrue(torch.isfinite(site.log_lr.grad))

    def test_delta_roundtrip_and_gradient(self):
        self.config.memory_kind='delta';model=SPORA(self.config)
        fast,velocity=model.initial_state(1,True)
        event=Event(torch.tensor([2]),torch.tensor([5]),torch.ones(1),'new')
        fast,velocity,_=write(model,fast,velocity,event,differentiable=True)
        model(event.key,fast).sum().backward()
        self.assertIsNotNone(model.memories[0].log_lr.grad)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'delta.pt';save_checkpoint(path,model,fast,velocity)
            restored,ff,_,_=load_checkpoint(path)
            with torch.no_grad():
                torch.testing.assert_close(model(event.key,fast),restored(event.key,ff),atol=0,rtol=0)

    def test_immediate_correction_has_gradient_before_repeat(self):
        from unittest.mock import patch
        import spora.p2 as protocol
        events,keys,_,_=stream(torch.Generator().manual_seed(18),1,8,self.config,'cpu',corrections=1)
        fast,velocity=self.model.initial_state(1,True)
        calls=[]
        original=protocol.F.cross_entropy
        def record(logits,targets,*args,**kwargs):
            calls.append(tuple(targets.shape))
            return original(logits,targets,*args,**kwargs)
        with patch.object(protocol.F,'cross_entropy',side_effect=record):
            world_loss(self.model,events,keys,fast,velocity,'one',.8)
        self.assertIn((1,),calls)

    def test_soft_projection_is_finite_and_respects_trust_gate(self):
        from spora.protection_p2 import protected_delta
        config=ModelConfig(width=32,layers=2,heads=2,memory_width=16,expert_rank=4,
                           memory_layers=(0,1),memory_kind='delta')
        model=SPORA(config)
        fast,velocity=model.initial_state(1)
        event=Event(torch.tensor([1]),torch.tensor([2]),torch.ones(1),'correction')
        protected=torch.tensor([[3,4,5]])
        updated,_=protected_delta(model,fast,velocity,event,protected)
        self.assertTrue(all(torch.isfinite(p).all() for site in updated for p in site))
        disabled=Event(event.key,event.value,torch.zeros(1),'untrusted')
        untouched,_=protected_delta(model,fast,velocity,disabled,protected)
        for old,new in zip(fast,untouched):
            for a,b in zip(old,new):torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_cli_multistep_commits_once_and_invalid_write_preserves_file(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'individual.pt'
            save_checkpoint(path,self.model,cursor=0)
            result=subprocess.run([sys.executable,'-m','spora.cli','--individual',str(path),
                                   'learn','1','2','--policy','two'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            _,_,_,data=load_checkpoint(path)
            self.assertEqual(data['cursor'],1)
            self.assertEqual(data['metadata']['last_inner_updates'],2)
            before=path.read_bytes()
            failed=subprocess.run([sys.executable,'-m','spora.cli','--individual',str(path),
                                   'learn','1','3','--threshold','1.5'],capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertEqual(before,path.read_bytes())
            self.assertFalse(path.with_suffix('.lock').exists())


if __name__=='__main__':unittest.main()
