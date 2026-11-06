import random,unittest
import torch
from spora.temporal import TemporalSPORA,TemporalConfig
from spora.growing import new_state
from spora.language_ladder import CortexLanguageBridge
from spora.anchor_query import extend_anchor,restore_anchor
from spora.language_training import interface_state,frozen_digest
from spora.language_bridge import encode
from scripts.query_task import TRAIN,DEV,TEST,wording,unknown,batch,prompt
from scripts.wording_task import TRAIN as SHORT_TRAIN
from scripts.ladder_task import tasks
from scripts.anchor_labels import target_positions
from scripts.train_query_anchor import objective

class AnchorTests(unittest.TestCase):
    def test_targets_point_to_question_not_record(self):
        row=tasks()['update']['train'][0]
        for template in list(SHORT_TRAIN)+[wording(t) for t in TRAIN+DEV+TEST]:
            for r in (row,unknown(row,random.Random(9))):
                pos=target_positions([r],[template])[0];tokens=encode(prompt(r,template))
                self.assertEqual(bytes(tokens[pos-2:pos+1]).decode('utf-8'),r['target'])
                record_end=len(('记录:'+''.join(f'{i+1}:{o}到{p};' for i,(o,p) in enumerate(r['events']))).encode('utf-8'))
                self.assertGreater(pos,record_end)

    def test_causality_gradient_identity_restore(self):
        torch.set_num_threads(2);torch.manual_seed(22)
        organism=TemporalSPORA(TemporalConfig(width=32,layers=2,heads=2,memory_width=64,expert_rank=4,memory_layers=(0,1),history_writer='rls',current_address='orthogonal'))
        state=new_state(organism);before=frozen_digest(organism,state)
        original=CortexLanguageBridge(organism,depth=4,local_bytes=True,evidence_reader=True,copy_reader=True,local_evidence=True,query_window=24)
        payload=dict(format='spora-language-ladder-v1',arm='cortex',depth=4,local_bytes=True,evidence_reader=True,copy_reader=True,local_evidence=True,query_window=24,organism_sha256='parent',state=interface_state(original))
        model=extend_anchor(organism,payload,'parent');r=tasks()['update']['train'][0]
        rows=[r,unknown(r,random.Random(9))];wordings=[wording(TRAIN[0])]*2;tokens,starts=batch(rows,wordings)
        with torch.no_grad():self.assertTrue(torch.equal(original(tokens,state),model(tokens,state)))
        optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=.001)
        objective(model,state,tokens,starts,rows,wordings).backward()
        self.assertGreater(float(model.anchor_q.weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.anchor_out.weight.grad.abs().sum()),0.)
        optimizer.step();self.assertEqual(before,frozen_digest(organism,state))
        changed=tokens.clone();changed[:,40:]=(changed[:,40:]+1)%256
        with torch.no_grad():
            self.assertTrue(torch.equal(model(tokens,state)[:,:40],model(changed,state)[:,:40]))
            self.assertTrue(torch.allclose(model(tokens,state).exp().sum(-1),torch.ones_like(tokens,dtype=torch.float),atol=1e-6))
        saved=dict(format='spora-anchor-query-v1',arm='context',organism_sha256='parent',state=interface_state(model))
        restored=restore_anchor(organism,saved,'parent')
        with torch.no_grad():self.assertTrue(torch.equal(model(tokens,state),restored(tokens,state)))
        with self.assertRaises(ValueError):restore_anchor(organism,saved,'other')

if __name__=='__main__':unittest.main()
