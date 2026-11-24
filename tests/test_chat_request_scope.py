import unittest

from spora.chat_request_scope import starts_new_topic, treats_quotes_as_text


class ChatRequestScopeTests(unittest.TestCase):
    def test_quoted_lookup_inside_text_operation(self):
        for question in (
            '只评价这句读起来是否自然，不要回答问题。“齐宁平常最爱喝什么？”',
            '把「齐宁平常最爱喝什么」换一种说法，只改写这句话。',
            '这句是否通顺？「何程的岗位是什么？」',
        ):
            with self.subTest(question=question):
                self.assertTrue(treats_quotes_as_text(question))

    def test_real_queries_and_unquoted_requests_keep_their_normal_path(self):
        for question in (
            '齐宁平常最爱喝什么？',
            '请回答「齐宁平常最爱喝什么？」',
            '请判断「齐宁最爱喝茶」是不是事实。',
            '说说齐宁最爱喝的“饮品”。',
            '先润色「何程的岗位是什么」，然后请回答。',
            '请润色「何程的岗位是什么」，也想知道何程实际上做什么。',
            '齐宁最爱喝什么？这条我还会翻译给别人。',
        ):
            with self.subTest(question=question):
                self.assertFalse(treats_quotes_as_text(question))

    def test_topic_switch_requires_a_user_instruction(self):
        self.assertTrue(starts_new_topic('换个话题。告诉我怎么准备午餐。'))
        self.assertTrue(starts_new_topic('另起一个话题，写一条短信。'))
        self.assertFalse(starts_new_topic('把“换个话题”翻成英文。'))
        self.assertFalse(starts_new_topic('说点别的例子，保持之前的要求。'))
