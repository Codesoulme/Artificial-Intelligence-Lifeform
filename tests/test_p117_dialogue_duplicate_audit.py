import unittest
from scripts.p117_dialogue_duplicate_audit import audit, reconstruct


def row(group, part, user, assistant):
    return (group,part,[{'role':'user','content':user},{'role':'assistant','content':assistant}])

class DuplicateAuditTests(unittest.TestCase):
    def test_original_families_and_first_prompt_transitivity(self):
        meta=[('a','first',True),('a','second',False),('b','second',True),('c','third',True)]
        selected,total=reconstruct(meta,limit=2)
        members=[set(x[2]) for x in selected]
        self.assertEqual(total,2)
        self.assertIn({0,1,2},members)
    def test_short_user_does_not_merge(self):
        result=audit([row('a','candidate','你好','甲'*35),row('b','development','你好','乙'*35)])
        self.assertEqual(result['counts']['distinct_cross_group_edges'],0)
        self.assertEqual(result['counts']['short_user_turn_occurrences_excluded'],2)
    def test_later_turn_and_original_group_partition(self):
        response='甲乙丙丁戊己庚辛壬癸'*5
        a=row('a','candidate','短问',response)
        b=('b','development',[{'role':'user','content':'不同问'}, {'role':'assistant','content':'不同答'}, {'role':'user','content':'后来问'}, {'role':'assistant','content':response}])
        c=row('c','quarantine','短问','完全不同'*10)
        x=audit([a,b,c]); self.assertEqual(x['edge_counts']['turn_exact_assistant'],1)
        self.assertEqual(x['counts']['candidate_development_edges'],1)
        self.assertEqual(x['counts']['original_groups_after_duplicate_union'],2)
    def test_near_and_role_boundary(self):
        s='天地玄黄宇宙洪荒日月盈昃辰宿列张'*5
        x=audit([row('a','candidate','甲',s),row('b','development','乙',s+'终')])
        self.assertEqual(x['edge_counts']['turn_near_assistant'],1)
        x=audit([row('a','candidate',s,'一'),row('b','development','短',s+'终')])
        self.assertEqual(x['counts']['distinct_cross_group_edges'],0)
    def test_output_aggregate_only_and_malformed(self):
        secret='独有秘密文字'*10
        x=audit([row('a','candidate','问',secret),('b','quarantine',[{'role':'assistant','content':secret},{}])])
        self.assertNotIn(secret,str(x))
        self.assertEqual(x['counts']['malformed_rows_excluded_from_text_comparison'],1)
        self.assertEqual(x['counts']['distinct_cross_group_edges'],0)
    def test_deterministic(self):
        rows=[row('a','candidate','一','答案'*30),row('b','development','二','答案'*30)]
        self.assertEqual(audit(rows),audit(list(reversed(rows))))

if __name__=='__main__': unittest.main()
