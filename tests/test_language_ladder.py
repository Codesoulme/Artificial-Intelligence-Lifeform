import copy,unittest
import torch
from spora.temporal import TemporalSPORA,TemporalConfig
from spora.growing import new_state
from spora.language_bridge import LanguageBridge
from spora.language_training import interface_state,frozen_digest,load_interface
from spora.language_ladder import CortexLanguageBridge,restore_ladder
from spora.short_reasoning import response_loss
from scripts.ladder_task import tasks,evidence_positions,matching_positions
from scripts.reasoning_task import prompt
from spora.language_bridge import encode


class LadderTests(unittest.TestCase):
    def test_local_evidence_variant(self):
        self.test_new_language_path_learns_without_changing_memory_and_restores(local_evidence=True,query_window=24)

    def test_new_language_path_learns_without_changing_memory_and_restores(self,local_evidence=False,query_window=32):
        torch.set_num_threads(2);torch.manual_seed(40)
        organism=TemporalSPORA(TemporalConfig(width=32,layers=2,heads=2,memory_width=64,
            expert_rank=4,memory_layers=(0,1),history_writer='rls',current_address='orthogonal'))
        state=new_state(organism);original=frozen_digest(organism,state)
        source={'format':'spora-byte-interface-v1','organism_sha256':'parent','state':interface_state(LanguageBridge(organism))}
        model=CortexLanguageBridge(organism,source,'parent',depth=4,local_bytes=True,evidence_reader=True,copy_reader=True,local_evidence=local_evidence,query_window=query_window)
        tokens=torch.randint(256,(2,16));changed=tokens.clone();changed[:,10:]=(changed[:,10:]+1)%256
        opt=torch.optim.AdamW(model.trainable_parameters(),lr=.001)
        opt.zero_grad();response_loss(model(tokens,state),tokens,torch.tensor([5,5])).backward()
        self.assertGreater(float(model.cortex[0].qkv.weight.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in organism.parameters()))
        opt.step();self.assertEqual(original,frozen_digest(organism,state))
        with torch.no_grad():self.assertTrue(torch.equal(model(tokens,state)[:,:10],model(changed,state)[:,:10]))
        payload={'format':'spora-language-ladder-v1','arm':'cortex','depth':4,'local_bytes':True,'evidence_reader':True,'copy_reader':True,'local_evidence':local_evidence,'query_window':query_window,'organism_sha256':'parent','state':interface_state(model)}
        restored=restore_ladder(copy.deepcopy(organism),payload,'parent')
        with torch.no_grad():self.assertTrue(torch.equal(model(tokens,state),restored(tokens,state)))
        with torch.no_grad():self.assertTrue(torch.allclose(model(tokens,state).exp().sum(-1),torch.ones_like(tokens,dtype=torch.float),atol=1e-6))
        with self.assertRaises(ValueError):load_interface(LanguageBridge(organism),payload,'parent')
        with self.assertRaises(ValueError):restore_ladder(copy.deepcopy(organism),payload,'other')
        for depth,local in ((2,False),(4,False),(4,True)):
            candidate=CortexLanguageBridge(copy.deepcopy(organism),depth=depth,local_bytes=local)
            snapshot={'format':'spora-language-ladder-v1','arm':'cortex','depth':depth,'local_bytes':local,
                'organism_sha256':'parent','state':interface_state(candidate)}
            back=restore_ladder(copy.deepcopy(organism),snapshot,'parent')
            self.assertEqual(set(interface_state(back)),set(snapshot['state']))
            self.assertTrue(all(torch.equal(t,interface_state(back)[k]) for k,t in snapshot['state'].items()))
        wrong=dict(payload,depth=2)
        with self.assertRaises(ValueError):restore_ladder(copy.deepcopy(organism),wrong,'parent')

    def test_ladder_pair_partitions_are_disjoint_and_labels_follow_last_event(self):
        data=tasks()
        self.assertEqual(len(data['single']['train']),24)
        for stage in ('select','update'):
            pairs={split:{tuple(r['pair']) for r in rows} for split,rows in data[stage].items()}
            self.assertFalse(pairs['train']&pairs['test'] or pairs['train']&pairs['development'] or pairs['test']&pairs['development'])
            for rows in data[stage].values():
                for row in rows:
                    self.assertEqual(row['answer'],[p for o,p in row['events'] if o==row['target']][-1])
                    position=evidence_positions([row])[0]
                    self.assertEqual(bytes(encode(prompt(row))[position-2:position+1]).decode('utf-8'),row['answer'])
                    matches=matching_positions([row])[0]
                    self.assertEqual(len(matches),len(row['indices']))
                    self.assertEqual(matches[-1],position)
                    self.assertEqual([bytes(encode(prompt(row))[p-2:p+1]).decode('utf-8') for p in matches],
                        [place for obj,place in row['events'] if obj==row['target']])
        expanded=tasks(expanded=True)
        for stage in ('select','update'):
            self.assertEqual(data[stage]['test'],expanded[stage]['test'])
            self.assertEqual(data[stage]['development'],expanded[stage]['development'])
            held={tuple(r['pair']) for s in ('development','test') for r in data[stage][s]}
            self.assertFalse(held & {tuple(r['pair']) for r in expanded[stage]['train']})
            self.assertGreater(len(expanded[stage]['train']),len(data[stage]['train']))


if __name__=='__main__':unittest.main()
