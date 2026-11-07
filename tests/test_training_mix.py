import json
import unittest
from spora.training_mix import choose_source


class TrainingMixTests(unittest.TestCase):
    def test_token_weighting_and_resume_with_different_batch_lengths(self):
        shares={'pretrain':.4,'teacher':.59,'foundation':.01}
        costs={'pretrain':8192,'teacher':1600,'foundation':600}
        counts={};kinds=[]
        for step in range(1000):
            kind=choose_source(shares,counts);kinds.append(kind)
            key=kind+'_source_targets';counts[key]=counts.get(key,0)+costs[kind]
            if step==432:
                split=json.loads(json.dumps(counts))
        total=sum(counts.values())
        for kind,share in shares.items():
            self.assertLess(abs(counts[kind+'_source_targets']/total-share),.005)
        self.assertLess(kinds.count('pretrain')/len(kinds),.2)
        for step in range(433,1000):
            kind=choose_source(shares,split)
            self.assertEqual(kind,kinds[step])
            key=kind+'_source_targets';split[key]=split.get(key,0)+costs[kind]
        self.assertEqual(counts,split)

    def test_invalid_mix_and_counts_fail(self):
        for shares in ({},{'bad':1},{'pretrain':0},{'pretrain':float('nan')},{'pretrain':.5}):
            with self.assertRaises(ValueError):choose_source(shares,{})
        with self.assertRaises(ValueError):choose_source({'teacher':1},{'teacher_source_targets':-1})


if __name__=='__main__':unittest.main()
