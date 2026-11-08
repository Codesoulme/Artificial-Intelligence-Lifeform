import json
import unittest

from scripts import sample_wildchat_private_review as review


def row(prompt, answer='这是一个普通回答。'):
    messages = [dict(role='user', content=prompt), dict(role='assistant', content=answer),
                dict(role='user', content='继续说明'), dict(role='assistant', content='好的。')]
    identity = review.digest(json.dumps(messages, ensure_ascii=False, sort_keys=True))
    return dict(messages=messages, identity=identity)


def grouped(*rows):
    group = review.digest('\n'.join(sorted(r['identity'] for r in rows)))
    for r in rows:
        r['group'] = group
    return group


class ReviewSampleTests(unittest.TestCase):
    def test_validates_complete_group_and_split(self):
        a, b = row('请解释这个话题', '第一个回答'), row('请解释这个话题', '第二个回答')
        group = grouped(a, b)
        # Avoid a synthetic group assigned to development.
        if int(group[:8], 16) % 10 == 0:
            b['messages'][1]['content'] += '。'
            b = row('请解释这个话题', b['messages'][1]['content'])
            group = grouped(a, b)
        self.assertNotEqual(int(group[:8], 16) % 10, 0)
        self.assertEqual(len(review.validate_groups([a, b])[group]), 2)
        with self.assertRaises(ValueError):
            review.validate_groups([a])

    def test_privacy_quarantines_whole_group_and_previous_review(self):
        a, b, c = row('主题甲'), row('主题乙', '联系我：13800138000'), row('主题丙')
        groups = {'a': [a, b], 'b': [c], 'c': [row('主题丁')]}
        chosen, sizes, excluded = review.select(groups, {'b'}, 1, 1)
        self.assertEqual(chosen[0][1], 'c')
        self.assertEqual(excluded['privacy_any_group'], 1)
        self.assertEqual(excluded['prior_review_group'], 1)
        self.assertEqual(sum(sizes.values()), 1)

    def test_deterministic_stratification_and_no_group_duplicates(self):
        groups = {}
        for i in range(8):
            r = row(f'主题{i}')
            if i % 2:
                r['messages'] += [dict(role='user', content='补充'), dict(role='assistant', content='收到')]
            groups[str(i)] = [r]
        first = review.select(groups, set(), 4, 42)
        self.assertEqual(first, review.select(groups, set(), 4, 42))
        self.assertEqual(len({g for _, g, _ in first[0]}), 4)
        self.assertEqual(len({s for s, _, _ in first[0]}), 2)

    def test_normalizes_fullwidth_contacts(self):
        self.assertIn('number_or_identifier', review.risks(row('号码 １３８００１３８０００')))


if __name__ == '__main__':
    unittest.main()
