import copy,random,unittest
import torch
from spora.temporal import TemporalSPORA,TemporalConfig
from spora.growing import new_state
from spora.language_ladder import CortexLanguageBridge
from spora.query_bridge import extend_query,restore_query
from spora.language_training import interface_state,frozen_digest
from scripts.query_task import TRAIN,DEV,TEST,wording,unknown,evaluation_rows,batch
from scripts.ladder_task import tasks
from scripts.train_query import objective

class QueryTests(unittest.TestCase):
    def test_extension_preserves_source_and_learns_causally(self):
        torch.set_num_threads(2);torch.manual_seed(22)
        organism=TemporalSPORA(TemporalConfig(width=32,layers=2,heads=2,memory_width=64,expert_rank=4,memory_layers=(0,1),history_writer='rls',current_address='orthogonal'))
        state=new_state(organism);before=frozen_digest(organism,state)
        source=CortexLanguageBridge(organism,depth=4,local_bytes=True,evidence_reader=True,copy_reader=True,local_evidence=True,query_window=24)
        payload=dict(format='spora-language-ladder-v1',arm='cortex',depth=4,local_bytes=True,evidence_reader=True,copy_reader=True,local_evidence=True,query_window=24,organism_sha256='parent',state=interface_state(source))
        model=extend_query(organism,payload,'parent')
        row=tasks()['update']['train'][0];absent=unknown(row,random.Random(8))
        tokens,starts=batch([row,absent],[wording(TRAIN[0])]*2)
        with torch.no_grad():self.assertTrue(torch.equal(source(tokens,state),model(tokens,state)))
        optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=.001)
        objective(model,state,tokens,starts,[row,absent]).backward()
        self.assertGreater(float(model.context_query[-1].weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in organism.parameters()))
        optimizer.step();self.assertEqual(before,frozen_digest(organism,state))
        changed=tokens.clone();changed[:,30:]=(changed[:,30:]+1)%256
        with torch.no_grad():
            self.assertTrue(torch.equal(model(tokens,state)[:,:30],model(changed,state)[:,:30]))
            self.assertTrue(torch.allclose(model(tokens,state).exp().sum(-1),torch.ones_like(tokens,dtype=torch.float),atol=1e-6))
        snapshot=dict(format='spora-context-query-v1',arm='context',organism_sha256='parent',state=interface_state(model))
        restored=restore_query(copy.deepcopy(organism),snapshot,'parent')
        with torch.no_grad():self.assertTrue(torch.equal(model(tokens,state),restored(tokens,state)))
        with self.assertRaises(ValueError):restore_query(organism,snapshot,'other')
        with self.assertRaises(ValueError):restore_query(organism,dict(snapshot,arm='local'),'parent')
        optimizer.zero_grad(set_to_none=True)
        tokens,starts=batch([absent],[wording(TRAIN[0])])
        loss=objective(model,state,tokens,starts,[absent]);self.assertTrue(torch.isfinite(loss));loss.backward()

    def test_partitions_unknown_and_context_bounds(self):
        self.assertEqual(len(set(TRAIN+DEV+TEST)),16)
        self.assertEqual([len(TRAIN),len(DEV),len(TEST)],[12,2,2])
        for split in ('development','test'):
            known,absent=evaluation_rows(split,96)
            self.assertEqual(len(known),len(absent))
            for a,b in zip(known,absent):
                self.assertEqual(a['events'],b['events'])
                self.assertNotIn(b['target'],[o for o,p in b['events']])
                self.assertEqual(b['answer'],'未知')
                for pair in TRAIN+DEV+TEST:
                    tokens,starts=batch([a,b],[wording(pair)]*2)
                    self.assertLess(tokens.shape[1]+16,256)

if __name__=='__main__':unittest.main()
