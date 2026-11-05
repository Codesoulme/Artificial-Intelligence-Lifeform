import copy,tempfile,unittest
from pathlib import Path
import torch
from spora.temporal import TemporalSPORA,TemporalConfig
from spora.growing import new_state,observe,grow,save,load,query
from spora.internal_memory import WriteControl,attach,observe_sourced,read_provenance,consolidation_request
from spora.state_codec import flatten,restore
from spora.durable import TensorStore
from spora.learned_read import LearnedReadPlan,prepare_router
from spora.memory_read import MemoryReadPlan
from spora.sleep_memory import propose
from spora.language_bridge import LanguageBridge


class InternalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(71)
        self.model=TemporalSPORA(TemporalConfig(width=32,layers=2,heads=2,memory_width=64,
            expert_rank=4,memory_layers=(0,1),history_writer='rls',current_address='orthogonal'))
        self.state=attach(self.model,new_state(self.model),WriteControl())

    def force_action(self,action):
        with torch.no_grad():
            for p in self.model.write_control.parameters():p.zero_()
            self.model.write_control.network[-1].bias[action]=20

    def test_hypotheses_cannot_write_current_even_if_network_requests(self):
        self.force_action(0)
        for source in ('inference','simulation'):
            state,result=observe_sourced(self.model,self.state,3,7,source,1.)
            self.assertEqual(result['action'],'defer');self.assertEqual(state.cursor,1)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(state.current,self.state.current)))
            self.assertFalse(torch.equal(state.pages[0][-1][0],self.state.pages[0][-1][0]))
            self.assertFalse(torch.equal(state.internal.provenance[0],self.state.internal.provenance[0]))

    def test_uncertain_writes_defer_and_legacy_input_cannot_bypass(self):
        self.force_action(4)
        with torch.no_grad():self.model.write_control.network[-1].bias.zero_()
        state,r=observe_sourced(self.model,self.state,3,7,'direct',1.)
        self.assertEqual(r['reason'],'uncertain_decision')
        with self.assertRaises(ValueError):observe(self.model,state,3,8,1)
        for quality in (float('nan'),-1,2):
            with self.assertRaises(ValueError):observe_sourced(self.model,state,3,8,'direct',quality)
        with self.assertRaises(ValueError):observe_sourced(self.model,state,3,8,'direct',1.,state.cursor+1)

    def test_old_source_unknown_and_no_read_mutation(self):
        self.state.internal.start_cursor=0
        before=flatten(self.model,self.state)[0]
        self.assertIsNone(read_provenance(self.model,self.state,3,0)['source'])
        self.assertEqual(flatten(self.model,self.state)[0],before)

    def test_state_roundtrip_growth_and_transaction(self):
        self.force_action(4)
        state,_=observe_sourced(self.model,self.state,3,7,'report',.4)
        state=grow(self.model,state)
        schema,tensors=flatten(self.model,state)
        restored,other=restore(schema,tensors)
        self.assertEqual(schema,flatten(restored,other)[0])
        self.assertEqual(len(other.internal.provenance),2)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.sqlite'
            with TensorStore.create(path,self.model,state) as store:
                changed,_=observe_sourced(self.model,state,4,8,'simulation',1.)
                def interrupt(stage):
                    if stage=='after_manifest':raise RuntimeError('interrupt')
                with self.assertRaises(RuntimeError):store.commit(self.model,changed,0,hook=interrupt)
                m,s,_,rev=store.read()
                self.assertEqual(rev,0);self.assertEqual(flatten(m,s)[0],schema)
                for name,t in flatten(m,s)[1].items():self.assertTrue(torch.equal(t,tensors[name]),name)
            portable=Path(temp)/'state.pt';save(portable,self.model,state);m,s,_=load(portable)
            self.assertEqual(flatten(m,s)[0],schema)

    def test_invalid_controller_state_cannot_be_saved(self):
        state=copy.deepcopy(self.state);state.internal.corrections=1
        with self.assertRaises(ValueError):flatten(self.model,state)
        state=copy.deepcopy(self.state);state.internal.router_cursor=False
        with self.assertRaises(ValueError):flatten(self.model,state)
        state=copy.deepcopy(self.state);state.internal.provenance[0].fill_(float('nan'))
        with self.assertRaises(ValueError):flatten(self.model,state)

    def test_legacy_continuous_encoding_loads_read_only(self):
        schema,tensors=flatten(self.model,self.state)
        schema['internal'].pop('encoding')
        tensors['internal/provenance/0']=tensors['internal/provenance/0'][:11]
        model,state=restore(schema,tensors)
        self.assertEqual(state.internal.encoding,1)
        with self.assertRaises(ValueError):observe_sourced(model,state,3,8,'direct',1.)

    def test_growth_and_sleep_keep_provenance_and_invalidate_router(self):
        self.force_action(4);state=self.state
        for i in range(64):state,_=observe_sourced(self.model,state,i,3,'simulation',.8)
        self.assertEqual(consolidation_request(self.model,state)['pages'],[])
        state.internal.informative=16
        self.assertEqual(consolidation_request(self.model,state)['pages'],[0])
        prepared,_=prepare_router(self.model,state,rank=4,steps=2)
        student,_=propose(self.model,prepared,0,rank=4,steps=1)
        self.assertIsNone(student.internal.router_factors)
        self.assertTrue(torch.equal(student.internal.provenance[0],prepared.internal.provenance[0]))
        self.assertIsNone(grow(self.model,prepared).internal.router_factors)

    def test_prepared_factors_persist_and_byte_interface_falls_back(self):
        prepared,_=prepare_router(self.model,self.state,rank=4,steps=3)
        model,state=restore(*flatten(self.model,prepared))
        bridge=LanguageBridge(model,read_mode='learned')
        tokens=torch.tensor([[256,19,20,257]])
        with torch.no_grad():
            out=bridge(tokens,state);bridge.read_mode='exact';expected=bridge(tokens,state)
        torch.testing.assert_close(out,expected,atol=1e-5,rtol=1e-5)
        bridge.read_mode='learned';state.internal=state.internal.invalidate_router()
        with torch.no_grad():bridge(tokens,state)
        self.assertTrue(all(x['router_unavailable'] for x in bridge.last_read_stats))


class RouterTests(unittest.TestCase):
    @torch.no_grad()
    def test_certificate_prunes_separated_and_falls_back_ambiguous(self):
        matrices=[torch.eye(16)[None]*(i+1) for i in range(4)]
        factors=[[torch.zeros(16,2),torch.zeros(2,16)] for _ in matrices]
        plan=LearnedReadPlan(matrices,factors);q=torch.zeros(3,16);q[:,0]=1
        keys=torch.zeros(4,16);keys[0,0]=200
        expected,_=MemoryReadPlan(matrices).read(q,keys)
        actual,stats=plan.read(q,keys)
        self.assertEqual(stats['pruned_queries'],3)
        self.assertLess(float((actual-expected).norm(dim=-1).max()),1e-4)
        _,ambiguous=plan.read(q,torch.zeros_like(keys))
        self.assertEqual(ambiguous['fallback_queries'],3)

    @torch.no_grad()
    def test_bad_factors_still_cannot_bypass_certificate(self):
        torch.manual_seed(8)
        matrices=[torch.randn(1,16,16) for _ in range(4)]
        factors=[[torch.randn(16,3)*10,torch.randn(3,16)*10] for _ in matrices]
        plan=LearnedReadPlan(matrices,factors);q=torch.randn(20,16);keys=torch.randn(4,16)
        actual,stats=plan.read(q,keys);expected,_=MemoryReadPlan(matrices).read(q,keys)
        self.assertEqual(stats['fallback_queries'],20)
        self.assertTrue(torch.equal(actual,expected))
        factors[0][0].add_(1)
        with self.assertRaises(ValueError):plan.read(q,keys)


if __name__=='__main__':unittest.main()
