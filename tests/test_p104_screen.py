import unittest
from scripts.p104_screen import assess, screen


def row(group, prompt='这是一个足够长的普通问题，请说说天气和日常活动。', answer='这是正常的回答，提供一些关于日常活动的建议。'):
    return {'group': group, 'messages': [
        {'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer},
        {'role': 'user', 'content': '谢谢，请继续详细解释这些安排。'},
        {'role': 'assistant', 'content': '可以根据时间和实际情况调整安排。'}]}


class ScreenTests(unittest.TestCase):
    def test_group_union_and_quarantine(self):
        rows = [row('a'), row('b'), row('b', prompt='另一个普通问题，想知道这些建议如何调整。', answer='联系 test@example.com')]
        counts, selected, groups = screen(rows)
        self.assertEqual(groups, 1)
        self.assertEqual(counts['groups_quarantined'], 1)
        self.assertEqual(selected, [])

    def test_split_is_group_stable_and_deterministic(self):
        rows = [row('a'), row('a', prompt='另一个问题，请详细说明日常安排的想法。'), row('b', prompt='不同的问题，请说明普通活动有哪些安排。')]
        a = screen(rows)
        self.assertEqual(a, screen(rows))
        self.assertEqual(a[1][0]['partition'], a[1][1]['partition'])

    def test_shape_and_sensitive_checks(self):
        self.assertEqual(assess({'messages': []}), ['shape'])
        self.assertIn('identity', assess(row('x', answer='我的身份证在这里。')))

    def test_single_turn_member_checked_but_not_selected(self):
        single = row('a')
        single['messages'] = single['messages'][:2]
        counts, selected, _ = screen([row('a'), single])
        self.assertEqual(len(selected), 1)
        self.assertEqual(counts['single_turn_rows_excluded'], 1)
        single['messages'][1]['content'] = '联系 test@example.com'
        self.assertEqual(screen([row('a'), single])[1], [])


if __name__ == '__main__':
    unittest.main()
