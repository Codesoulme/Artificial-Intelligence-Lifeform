import unittest,torch
from spora.temporal import TemporalSPORA,TemporalConfig
from spora.growing import new_state,observe,grow,GrowingState
from spora.language_bridge import LanguageBridge,causal_loss,encode,decode,BOS,EOS

class LanguageInterfaceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(95)
        self.organism=TemporalSPORA(TemporalConfig(width=32,layers=2,heads=2,memory_width=16,expert_rank=4,memory_layers=(0,1),history_writer='rls'))
        self.state=observe(self.organism,new_state(self.organism),2,5,1)
        self.bridge=LanguageBridge(self.organism)

    def test_utf8_roundtrip_only_no_language_training(self):
        for text in ('hello','中文字符🙂',''):
            self.assertEqual(decode(encode(text)),text)

    def test_future_tokens_do_not_change_prefix(self):
        tokens=torch.randint(0,256,(2,12));changed=tokens.clone();changed[:,7:]=(changed[:,7:]+1)%256
        with torch.no_grad():a=self.bridge(tokens,self.state);b=self.bridge(changed,self.state)
        torch.testing.assert_close(a[:,:7],b[:,:7],atol=0,rtol=0)

    def test_random_byte_step_preserves_organism_and_memories(self):
        snapshot={n:p.detach().clone() for n,p in self.organism.named_parameters()}
        memories=[p.detach().clone() for p in self.state.current]
        tokens=torch.randint(0,256,(2,12));optimizer=torch.optim.AdamW(self.bridge.trainable_parameters(),lr=1e-3)
        before=self.bridge.byte_embedding.weight.detach().clone()
        loss=causal_loss(self.bridge(tokens,self.state),tokens);loss.backward()
        self.assertIsNotNone(self.bridge.query_projection.weight.grad)
        self.assertGreater(float(self.bridge.query_projection.weight.grad.norm()),0)
        optimizer.step()
        self.assertFalse(torch.equal(before,self.bridge.byte_embedding.weight))
        for n,p in self.organism.named_parameters():torch.testing.assert_close(p,snapshot[n],atol=0,rtol=0)
        for a,b in zip(memories,self.state.current):torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_interface_actually_reads_memory(self):
        tokens=torch.randint(0,256,(1,9))
        with torch.no_grad():a=self.bridge(tokens,self.state);b=self.bridge(tokens,new_state(self.organism))
        self.assertFalse(torch.equal(a,b))

    def test_page_position_is_visible_to_untrained_interface(self):
        state=grow(self.organism,self.state)
        swapped=GrowingState(state.current,[state.pages[1],state.pages[0]],state.cursor)
        tokens=torch.randint(0,256,(1,9))
        with torch.no_grad():a=self.bridge(tokens,state);b=self.bridge(tokens,swapped)
        self.assertFalse(torch.equal(a,b))

if __name__=='__main__':unittest.main()
