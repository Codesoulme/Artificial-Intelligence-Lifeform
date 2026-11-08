import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from p117_dialogue_link_review import links, plan


def rec(g,p,line,u,a):
    return (g,p,line,[{'role':'user','content':u},{'role':'assistant','content':a}])

class LinkReviewTests(unittest.TestCase):
    def test_generic_answer_not_shared_meaning(self):
        x=links([rec('a','candidate',1,'甲','好的'),rec('b','development',2,'乙','好的')])
        self.assertEqual(x[0]['judgment'],'generic_response_only')
        self.assertNotIn('好的',str(x))
        self.assertEqual(x[0]['evidence'][0]['source_lines'],[1,2])
    def test_shared_pair_and_uncertainty(self):
        u='一个比较长的用户问题包含不少明确文字和额外具体约束条件'; a='这里给出比较长而且有很多细节的完整应答包含具体上下文信息'
        x=links([rec('a','candidate',1,u,a),rec('b','development',2,u,a)])
        self.assertEqual(x[0]['judgment'],'shared_prompt_and_nontrivial_answer')
        y=links([rec('a','candidate',1,u,'甲'),rec('b','development',2,u,'乙')])
        self.assertEqual(y[0]['judgment'],'shared_turn_semantics_uncertain')
    def test_short_questions_no_edge(self):
        self.assertEqual(links([rec('a','candidate',1,'你好','甲'),rec('b','development',2,'你好','乙')]),[])
    def test_transitive_quarantine_and_development_never_to_candidate(self):
        states={'a':'candidate','b':'development','c':'candidate','d':'quarantine','e':'candidate'}
        edges=[{'groups':['a','b']},{'groups':['c','d']},{'groups':['d','e']}]
        p={x['group_sha256']:x for x in plan(states,edges)}
        self.assertEqual(p['a']['proposed'],'quarantine')
        self.assertEqual(p['b']['proposed'],'development')
        self.assertEqual({p[x]['proposed'] for x in 'cde'},{'quarantine'})
        self.assertEqual(p['c']['family_sha256'],p['e']['family_sha256'])
        self.assertEqual(plan(states,edges),plan(dict(reversed(list(states.items()))),list(reversed(edges))))

if __name__=='__main__': unittest.main()
