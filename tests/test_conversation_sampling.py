import unittest,torch
from spora.conversation_sampling import nucleus_probabilities

class SamplingTests(unittest.TestCase):
    def test_nucleus_keeps_crossing_token_and_forbidden_stays_zero(self):
        scores=torch.tensor([.5,.3,.2,0.]).log()
        p=nucleus_probabilities(scores,1.,.7)
        self.assertTrue(torch.allclose(p,torch.tensor([.625,.375,0.,0.])))
        self.assertTrue(torch.allclose(nucleus_probabilities(scores,1.,1.),torch.tensor([.5,.3,.2,0.])))
        self.assertTrue(torch.equal(nucleus_probabilities(scores,1.,.01),torch.tensor([1.,0.,0.,0.])))
        for temperature,top_p in ((0,.9),(1,0),(1,1.1)):
            with self.assertRaises(ValueError):nucleus_probabilities(scores,temperature,top_p)

if __name__=='__main__':unittest.main()
