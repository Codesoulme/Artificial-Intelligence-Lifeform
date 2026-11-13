import json
import unittest

from scripts.audit_p236_duconv import conversation_key
from scripts.deepseek_spora_duconv_span_audit import person_names, select_candidates


class DuConvSpanAuditTests(unittest.TestCase):
    def test_names_need_supported_relation_and_plain_chinese(self):
        knowledge = [['小林', '性别', '女'], ['电影', '主演', '小周'],
                     ['作品', '类型', '动画'], ['电影', '主演', '小周,小林']]
        self.assertEqual(person_names(knowledge), {'小林', '小周'})

    def test_excludes_previously_reviewed_and_multi_match_turns(self):
        excluded_conversation = ['请问小林今天喜欢喝什么饮料？']
        valid_conversation = ['你知道小周是这部作品的导演吗？', '小林和小周都在演这部电影吗？']
        knowledge = [['小林', '性别', '女'], ['电影', '导演', '小周']]
        rows = [json.dumps({'conversation': excluded_conversation, 'knowledge': knowledge}),
                json.dumps({'conversation': valid_conversation, 'knowledge': knowledge})]
        chosen = select_candidates(rows, {conversation_key(excluded_conversation)}, count=1)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]['name'], '小周')
        self.assertEqual(chosen[0]['turn'], 0)


if __name__ == '__main__':
    unittest.main()
