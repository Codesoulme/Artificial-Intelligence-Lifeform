import unittest,copy
import torch
from spora.temporal import TemporalSPORA,TemporalConfig
from spora.growing import new_state
from spora.conversation import ConversationBridge,ConversationConfig,restore_conversation,generate
from spora.language_training import interface_state,frozen_digest

class ConversationTests(unittest.TestCase):
    def test_causality_incremental_memory_and_restore(self):
        torch.set_num_threads(2);torch.manual_seed(17)
        organism=TemporalSPORA(TemporalConfig(width=32,layers=2,heads=2,memory_width=64,expert_rank=4,memory_layers=(0,1),history_writer='rls',current_address='orthogonal'))
        state=new_state(organism);before=frozen_digest(organism,state)
        config=ConversationConfig(vocabulary=64,width=32,depth=2,heads=4,hidden=64,context=64,memory_layers=(0,1))
        model=ConversationBridge(organism,config);self.assertEqual(before,frozen_digest(organism,state))
        tokens=torch.randint(8,64,(2,16));changed=tokens.clone();changed[:,8:]=(changed[:,8:]+1)%64
        with torch.no_grad():
            self.assertTrue(torch.equal(model(tokens,state)[:,:8],model(changed,state)[:,:8]))
            complete=model(tokens,state);_,past=model(tokens[:,:7],state,use_past=True)
            remainder,past=model(tokens[:,7:11],state,past=past,use_past=True)
            self.assertTrue(torch.allclose(complete[:,7:11],remainder,atol=1e-6))
            for i in range(11,16):
                step,past=model(tokens[:,i:i+1],state,past=past,use_past=True)
                self.assertTrue(torch.allclose(complete[:,i:i+1],step,atol=1e-6))
        opt=torch.optim.AdamW(model.trainable_parameters(),lr=.001)
        logits=model(tokens,state);torch.nn.functional.cross_entropy(logits[:,:-1].flatten(0,1),tokens[:,1:].flatten()).backward()
        self.assertTrue(all(p.grad is None for p in organism.parameters()))
        self.assertGreater(float(model.blocks[0].qkv.weight.grad.abs().sum()),0)
        opt.step();self.assertEqual(before,frozen_digest(organism,state))
        from dataclasses import asdict
        payload=dict(format='spora-conversation-v1',organism_sha256='parent',tokenizer_sha256='vocab',config=asdict(config),state=interface_state(model))
        restored=restore_conversation(copy.deepcopy(organism),payload,'parent','vocab')
        with torch.no_grad():self.assertTrue(torch.equal(model(tokens,state),restored(tokens,state)))
        with self.assertRaises(ValueError):restore_conversation(organism,payload,'parent','other')
        with self.assertRaises(ValueError):restore_conversation(organism,payload,'other','vocab')
        self.assertEqual(generate(model,state,[2,3,4],5,8),generate(restored,state,[2,3,4],5,8))

if __name__=='__main__':unittest.main()
