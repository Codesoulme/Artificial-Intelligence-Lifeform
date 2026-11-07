import copy
import unittest
import torch
from torch.nn import functional as F
from spora.token_objective import backward_token_budget


class TokenObjectiveTests(unittest.TestCase):
    def test_different_source_lengths_match_single_token_mean_and_update(self):
        torch.manual_seed(28)
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.embedding=torch.nn.Embedding(12,8);self.head=torch.nn.Linear(8,12)
            def forward(self,tokens,state):return self.head(self.embedding(tokens))
        model=Tiny().double();reference=copy.deepcopy(model)
        batches=[]
        for length in (3,11,2):
            x=torch.randint(0,12,(2,length));y=torch.randint(0,12,(2,length));y[0,0]=-100
            batches.append((x,y))
        # Independent reference concatenates only supervised outputs, preserving
        # each token's weight even when sources have different sequence lengths.
        logits=[];labels=[]
        for x,y in batches:
            selected=y!=-100;logits.append(reference(x,None)[selected]);labels.append(y[selected])
        expected=F.cross_entropy(torch.cat(logits),torch.cat(labels));expected.backward()
        actual,total=backward_token_budget(model,None,batches)
        self.assertEqual(total,sum(len(y) for y in labels))
        self.assertAlmostEqual(float(expected),float(actual),places=6)
        for a,b in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=1e-12,rtol=1e-12)
        optimizers=[torch.optim.AdamW(m.parameters(),lr=.001) for m in (model,reference)]
        for optimizer in optimizers:optimizer.step()
        for a,b in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)

    def test_no_supervision_rejected(self):
        model=torch.nn.Linear(2,2)
        with self.assertRaises(ValueError):backward_token_budget(model,None,[])
        x=torch.ones(2,3,dtype=torch.long)
        with self.assertRaises(ValueError):backward_token_budget(model,None,[(x,torch.full_like(x,-100))])


if __name__=='__main__':unittest.main()
