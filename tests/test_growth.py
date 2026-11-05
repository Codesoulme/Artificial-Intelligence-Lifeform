import tempfile,unittest
from pathlib import Path
import torch
from spora.temporal import TemporalSPORA,TemporalConfig
from spora.growing import new_state,grow,observe,query,save,load,scalars,migrate
from spora.checkpoint import save_checkpoint
from spora.growing_cli import mutate

class GrowthTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(91)
        self.model=TemporalSPORA(TemporalConfig(width=32,layers=2,heads=2,memory_width=16,expert_rank=4,memory_layers=(0,1),history_writer='rls'))

    def test_growth_preserves_all_prior_state_and_queries(self):
        state=observe(self.model,new_state(self.model),2,5,1)
        k=torch.tensor([2,3]);m=torch.tensor([1,0]);t=torch.tensor([0,0])
        before=query(self.model,state,k,m,t);larger=grow(self.model,state)
        torch.testing.assert_close(before,query(self.model,larger,k,m,t),atol=0,rtol=0)
        self.assertIs(state.pages[0],larger.pages[0]);self.assertGreater(scalars(larger)['total'],scalars(state)['total'])
        self.assertEqual(state.cursor,larger.cursor)

    def test_capacity_and_duplicate_time_reject(self):
        state=new_state(self.model)
        with self.assertRaises(ValueError):grow(self.model,state,pages=20,max_pages=100)
        with self.assertRaises(ValueError):observe(self.model,state,2,5,1,time=1)
        state.cursor=64
        with self.assertRaises(ValueError):observe(self.model,state,2,5,1)
        state=grow(self.model,state);updated=observe(self.model,state,2,5,1,time=64)
        self.assertEqual(updated.cursor,65)
        for a,b in zip(state.pages[0],updated.pages[0]):
            for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=0,rtol=0)

    def test_restart_and_invalid_save_preserves_file(self):
        state=grow(self.model,observe(self.model,new_state(self.model),2,5,1))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'growing.pt';save(path,self.model,state);model,other,_=load(path)
            a=observe(self.model,state,2,9,1);b=observe(model,other,2,9,1)
            for x,y in zip(a.current,b.current):torch.testing.assert_close(x,y,atol=0,rtol=0)
            prior=path.read_bytes()
            with torch.no_grad():a.current[0].fill_(float('nan'))
            with self.assertRaises(ValueError):save(path,self.model,a)
            self.assertEqual(prior,path.read_bytes())

    def test_adoption_preserves_live_temporal_function(self):
        f,v=self.model.initial_state(1)
        f,v=self.model.write_temporal(f,v,torch.tensor([2]),torch.tensor([5]),torch.tensor([0]),torch.ones(1))
        with tempfile.TemporaryDirectory() as d:
            old=Path(d)/'old.pt';new=Path(d)/'new.pt';save_checkpoint(old,self.model,f,v,cursor=1)
            original=old.read_bytes();adopted,state=migrate(old,new)
            keys=torch.tensor([2,2,3]);modes=torch.tensor([0,1,1]);times=torch.tensor([0,0,1])
            expected=self.model.predict_many(keys[None],f,modes[None],times[None]).squeeze(0)
            torch.testing.assert_close(expected,query(adopted,state,keys,modes,times),atol=0,rtol=0)
            self.assertEqual(original,old.read_bytes())
            with self.assertRaises(FileExistsError):migrate(old,new)

    def test_cli_invalid_writes_preserve_file_and_lock(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'person.pt';save(p,self.model,new_state(self.model));original=p.read_bytes()
            with self.assertRaises(ValueError):mutate(p,'learn',key=1,value=16,trusted=1)
            self.assertEqual(original,p.read_bytes())
            lock=p.with_suffix('.lock');lock.write_text('another writer')
            with self.assertRaises(FileExistsError):mutate(p,'grow',pages=1)
            self.assertTrue(lock.exists());self.assertEqual(original,p.read_bytes())

if __name__=='__main__':unittest.main()
