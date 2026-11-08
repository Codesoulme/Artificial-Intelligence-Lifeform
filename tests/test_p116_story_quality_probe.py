import unittest
import numpy as np
from scripts.p116_story_quality_probe import groups, split_groups, metrics, predict_model, rule

class ProbeTest(unittest.TestCase):
    def test_groups_and_stratification(self):
        texts = [('甲乙丙丁戊己庚辛' * 15) + str(i) for i in range(2)] + [str(i)*120 for i in range(10)]
        g = groups(texts)
        self.assertEqual(g[0],g[1])
        y = np.array([1,1]+[i%2 for i in range(10)])
        f = split_groups(y,g,folds=2)
        self.assertEqual(f[0],f[1])
        self.assertEqual(set(f),{0,1})
        for k in range(2): self.assertEqual(set(y[f==k]),{0,1})
    def test_counts_and_zero_denominators(self):
        m = metrics([1,1,0,0],[1,0,1,0])
        self.assertEqual([m[k] for k in ('tp','fn','fp','tn')],[1,1,1,1])
        self.assertEqual(m['retention']['rate'],0.5)
        self.assertIsNone(metrics([0],[0])['recall']['rate'])
    def test_model_and_rule(self):
        x = np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
        self.assertEqual(predict_model(x,np.array([1,1,0,0]),x).tolist(),[1,1,0,0])
        self.assertEqual(rule('短'),1)

if __name__ == '__main__': unittest.main()
