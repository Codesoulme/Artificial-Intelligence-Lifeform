import unittest
import torch
import torch.nn.functional as F
from scripts.deepseek_learned_kernel_probe_20261117 import LearnedKernel


class LearnedKernelTests(unittest.TestCase):
    def test_aggregated_read_matches_pairwise_weights_and_has_gradients(self):
        torch.manual_seed(41)
        m=LearnedKernel()
        q,k,v=torch.randn(4,128),torch.randn(9,128),torch.randn(9,128)
        qf=F.softplus(m.qmap(F.layer_norm(q,(128,))))+1e-5
        kf=F.softplus(m.kmap(F.layer_norm(k,(128,))))+1e-5
        weights=qf@kf.T
        expected=(weights@v)/weights.sum(-1,keepdim=True)
        result=m(q,k,v)
        torch.testing.assert_close(result,expected,atol=1e-6,rtol=1e-5)
        combined_a=torch.zeros(128,128)
        combined_z=torch.zeros(128)
        for start in range(0,len(k),3):
            part=kf[start:start+3]
            combined_a=combined_a+part.T@v[start:start+3]
            combined_z=combined_z+part.sum(0)
        chunk=(qf@combined_a)/(qf@combined_z).unsqueeze(-1)
        torch.testing.assert_close(result,chunk,atol=1e-6,rtol=1e-5)
        result.square().sum().backward()
        for p in m.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())


if __name__=='__main__':unittest.main()
