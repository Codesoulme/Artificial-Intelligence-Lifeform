import random
import unittest
from spora.research_hard_identity_episode import hard_episode


class HardIdentityEpisodeTests(unittest.TestCase):
    def test_positive_and_two_negative_forms(self):
        names=('甲人','乙人','丙人','丁人','戊人')
        rng=random.Random(7)
        query,records,target=hard_episode('甲人',names,True,0,rng)
        self.assertEqual(records[target],query)
        self.assertIn('甲人甲',records)
        query,records,target=hard_episode('甲人',names,False,1,rng)
        self.assertEqual(query,'甲人甲');self.assertIn('甲人',records)
        self.assertNotIn(query,records);self.assertEqual(target,4)
        query,records,target=hard_episode('甲人',names,False,3,rng)
        self.assertEqual(query,'甲人');self.assertIn('甲人甲',records)
        self.assertNotIn(query,records);self.assertEqual(target,4)
