import tempfile
from pathlib import Path
import unittest
import torch
from spora.model import SPORA, ModelConfig
from spora.checkpoint import save_checkpoint, load_checkpoint
from spora.tasks import absorb


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.model=SPORA(ModelConfig(width=32, layers=2, heads=2, memory_width=16,
                                     expert_rank=4,memory_layers=(0,1)))
        self.keys=torch.tensor([[1,5,9],[2,7,10]])
        self.values=torch.tensor([[3,4,5],[7,8,9]])

    def test_outer_gradients_reach_write_rule(self):
        fast,_=absorb(self.model,self.keys,self.values,differentiable=True)
        loss=torch.nn.functional.cross_entropy(self.model.predict_many(self.keys,fast).flatten(0,1),self.values.flatten())
        loss.backward()
        for site in self.model.memories:
            self.assertIsNotNone(site.log_lr.grad)
            self.assertTrue(torch.isfinite(site.log_lr.grad))
            self.assertGreater(abs(site.log_lr.grad.item()),1e-9)
            self.assertGreater(site.key.weight.grad.norm().item(),0)

    def test_only_fast_parameters_change_during_experience(self):
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        initial,_=self.model.initial_state(2)
        fast,_=absorb(self.model,self.keys,self.values)
        self.assertTrue(all(torch.equal(v,self.model.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(any(not torch.equal(a,b) for sa,sb in zip(initial,fast) for a,b in zip(sa,sb)))
        snapshot=[[p.clone() for p in site] for site in fast]
        with torch.no_grad(): self.model.predict_many(self.keys,fast)
        self.assertTrue(all(torch.equal(a,b) for sa,sb in zip(snapshot,fast) for a,b in zip(sa,sb)))

    def test_untrusted_write_is_noop(self):
        fast,velocity=absorb(self.model,self.keys,self.values)
        after,momentum=self.model.write_event(fast,velocity,self.keys[:,0],self.values[:,0],trusted=torch.zeros(2))
        for left,right in [(fast,after),(velocity,momentum)]:
            self.assertTrue(all(torch.equal(a,b) for sa,sb in zip(left,right) for a,b in zip(sa,sb)))

    def test_roundtrip_and_continuation(self):
        fast,velocity=absorb(self.model,self.keys,self.values)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'individual.pt'
            save_checkpoint(path,self.model,fast,velocity,cursor=3)
            restored,rf,ru,data=load_checkpoint(path)
            self.assertEqual(data['cursor'],3)
            self.assertNotIn('keys',data)
            self.assertNotIn('values',data)
            with torch.no_grad():
                torch.testing.assert_close(self.model.predict_many(self.keys,fast),restored.predict_many(self.keys,rf),rtol=0,atol=0)
            next_fast,_=self.model.write_event(fast,velocity,self.keys[:,0],self.values[:,1])
            resumed,_=restored.write_event(rf,ru,self.keys[:,0],self.values[:,1])
            for left,right in zip(next_fast,resumed):
                for a,b in zip(left,right): torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_causal_inputs_and_memory_sensitivity(self):
        tokens=self.model.query_tokens(self.keys[:,0])
        self.assertFalse(bool((tokens>=self.model.config.value_start).any()))
        fast1,_=absorb(self.model,self.keys,self.values)
        fast2,_=absorb(self.model,self.keys,(self.values+1)%self.model.config.value_count)
        self.assertFalse(torch.equal(self.model.predict_many(self.keys,fast1),self.model.predict_many(self.keys,fast2)))

    def test_nonfinite_save_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'valid.pt'
            save_checkpoint(path,self.model)
            original=path.read_bytes()
            with torch.no_grad(): next(self.model.parameters()).fill_(float('nan'))
            with self.assertRaises(ValueError): save_checkpoint(path,self.model)
            self.assertEqual(path.read_bytes(),original)

    def test_worlds_are_independent(self):
        together,_=absorb(self.model,self.keys,self.values)
        alone,_=absorb(self.model,self.keys[:1],self.values[:1])
        for batch,single in zip(together,alone):
            for a,b in zip(batch,single):
                torch.testing.assert_close(a[:1],b,rtol=1e-5,atol=1e-6)

    def test_reset_removes_experience_but_preserves_backbone(self):
        fast,velocity=absorb(self.model,self.keys,self.values)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'individual.pt'
            save_checkpoint(path,self.model,fast,velocity,cursor=3)
            restored,reset,_,_=load_checkpoint(path,reset_fast=True)
            initial,_=self.model.initial_state(2)
            with torch.no_grad():
                torch.testing.assert_close(self.model.predict_many(self.keys,initial),
                                           restored.predict_many(self.keys,reset),rtol=0,atol=0)

    def test_fresh_process_without_answers(self):
        from spora.evaluate import isolated_query
        keys,values=self.keys[:1],self.values[:1]
        fast,velocity=absorb(self.model,keys,values)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'individual.pt'
            save_checkpoint(path,self.model,fast,velocity,cursor=3)
            result=isolated_query(path,keys.flatten().tolist())
            with torch.no_grad():
                expected=self.model.predict_many(keys,fast).argmax(-1).flatten().tolist()
            self.assertEqual(result['predictions'],expected)


if __name__=='__main__':
    unittest.main()
