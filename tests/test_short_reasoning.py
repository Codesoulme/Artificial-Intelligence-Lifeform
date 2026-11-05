import copy,unittest
import torch
from spora.temporal import TemporalSPORA,TemporalConfig
from spora.growing import new_state
from spora.language_bridge import LanguageBridge,VOCAB,encode
from spora.language_training import interface_state,load_interface,frozen_digest
from spora.short_reasoning import RecurrentLanguageBridge,response_loss,generate_equal_prefixes,restore_research_interface
from scripts.reasoning_task import dataset,label,batch,balanced,changed_query,reversed_events,score


class ShortReasoningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(36)
        self.organism=TemporalSPORA(TemporalConfig(width=32,layers=2,heads=2,memory_width=64,
            expert_rank=4,memory_layers=(0,1),history_writer='rls',current_address='orthogonal'))
        self.state=new_state(self.organism)
        self.base=LanguageBridge(self.organism)
        self.source={'format':'spora-byte-training-v2','organism_sha256':'parent','state':interface_state(self.base)}

    def test_world_splits_do_not_overlap_and_counterfactual_answers_change(self):
        data=dataset();pairs={split:{tuple(row['pair']) for row in rows} for split,rows in data.items()}
        self.assertFalse(pairs['train']&pairs['development'] or pairs['train']&pairs['test'] or pairs['development']&pairs['test'])
        for rows in data.values():
            for row in rows:
                self.assertEqual(row['answer'],row['events'][max(row['indices'])-1][1])
                self.assertNotEqual(row['answer'],changed_query(row)['answer'])
                if len(row['indices'])==2:self.assertNotEqual(row['answer'],reversed_events(row)['answer'])
        selected=balanced(data['test'],96,2820000)
        self.assertEqual([sum(r['answer']==p for r in selected) for p in ('桌','柜','箱','架')],[24]*4)

    def test_response_loss_excludes_prompt_and_padding(self):
        tokens=torch.tensor([[256,10,11,12,257,258]])
        logits=torch.zeros(1,6,VOCAB,requires_grad=True)
        loss=response_loss(logits,tokens,torch.tensor([3]));loss.backward()
        self.assertEqual(float(logits.grad[:,:2].abs().sum()),0.)
        self.assertGreater(float(logits.grad[:,2:4].abs().sum()),0.)
        self.assertEqual(float(logits.grad[:,4:].abs().sum()),0.)

    def test_new_module_starts_as_identity_and_keeps_causal_prefix(self):
        model=RecurrentLanguageBridge(copy.deepcopy(self.organism),2,self.source,'parent')
        tokens=torch.randint(256,(2,12));changed=tokens.clone();changed[:,7:]=(changed[:,7:]+1)%256
        with torch.no_grad():
            self.assertTrue(torch.equal(self.base(tokens,self.state),model(tokens,self.state)))
            self.assertTrue(torch.equal(model(tokens,self.state)[:,:7],model(changed,self.state)[:,:7]))
        before=frozen_digest(model.organism,self.state)
        optimizer=torch.optim.Adam(model.trainable_parameters(),lr=.001)
        for _ in range(2):
            optimizer.zero_grad();response_loss(model(tokens,self.state),tokens,torch.tensor([3,3])).backward();optimizer.step()
        self.assertEqual(before,frozen_digest(model.organism,self.state))
        self.assertGreater(float(model.reasoner.qkv.weight.grad.abs().sum()),0.)
        with torch.no_grad():self.assertTrue(torch.equal(model(tokens,self.state)[:,:7],model(changed,self.state)[:,:7]))

    def test_new_interface_restores_exactly_and_legacy_loader_rejects_it(self):
        model=RecurrentLanguageBridge(copy.deepcopy(self.organism),2,self.source,'parent')
        payload={'format':'spora-short-reasoning-v1','arm':'recurrent_trace','passes':2,
            'organism_sha256':'parent','state':interface_state(model)}
        restored=restore_research_interface(copy.deepcopy(self.organism),payload,'parent')
        tokens=torch.randint(256,(1,10))
        with torch.no_grad():self.assertTrue(torch.equal(model(tokens,self.state),restored(tokens,self.state)))
        with self.assertRaises(ValueError):load_interface(self.base,payload,'parent')
        with self.assertRaises(ValueError):restore_research_interface(copy.deepcopy(self.organism),payload,'different')

    def test_evidence_and_final_answer_are_scored_separately(self):
        row={'events':[('杯','柜'),('球','架'),('杯','桌')],'target':'杯'};row.update(label(row['events'],row['target']))
        output={'text':'据:2;后:2;答:桌。','utf8_valid':True,'ended_with_eos':True}
        scored=score(row,output,'trace')
        self.assertTrue(scored['answer_correct']);self.assertFalse(scored['trace_correct']);self.assertFalse(scored['evidence_correct'])
        output['text']=row['trace'];self.assertTrue(score(row,output,'trace')['trace_correct'])
        output['ended_with_eos']=False;self.assertFalse(score(row,output,'trace')['trace_correct'])

    def test_batched_raw_generation_equals_single_sequence(self):
        prefixes=[encode('abc')[:-1],encode('xyz')[:-1]]
        batched=generate_equal_prefixes(self.base,self.state,prefixes,6)
        for prefix,actual in zip(prefixes,batched):
            self.assertEqual(actual,generate_equal_prefixes(self.base,self.state,[prefix],6)[0])


if __name__=='__main__':unittest.main()
