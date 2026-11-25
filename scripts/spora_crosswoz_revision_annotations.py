"""Reviewed turn references in the pinned CrossWOZ validation split.

Acceptance is scoped to the chosen plan. It does not change a person's global
preferences. The two states use the same evidence spans and different prefixes.
"""

# Dialogue, first user turn, field, original span, system proposal span.
# The intervening system turn presents the fallback. The next user turn accepts it.
ACCEPTED = (
    ('9000', 4, '游玩时长', '0.3小时 - 0.5小时', '0.5小时 - 1小时'),
    ('211', 22, '人均消费', '150-500元', '67元'),
    ('10071', 0, '酒店类型', '经济型', '高档型'),
    ('2365', 4, '人均消费', '50-100元', '17元'),
    ('8787', 2, '人均消费', '150-500元', '42元'),
    ('7804', 22, '门票价格', '100-150元', '70元'),
    ('7633', 10, '酒店评分', '5分', '4.6分'),
    ('10976', 0, '门票价格', '50-100元', '10元'),
    ('10611', 0, '人均消费', '1000元以上', '53元'),
    ('6864', 0, '酒店评分', '5分', '4.9分'),
    ('10446', 0, '人均消费', '50-100元', '138元'),
)

# Dialogue, included turns, two (turn, span) references, active spans, rationale.
# The messages are looked up at run time and are not embedded in the manifest.
CONTROLS = (
    ('information_only', '10611', (8, 9, 10), ((8, '5分'), (9, '4.3分')), (0,),
     '只问推荐的是哪一家，尚未接受降到4.3分。'),
    ('information_only', '9388', (2, 3, 4), ((2, '舒适型'), (3, '豪华型')), (0,),
     '索要电话询问优惠，尚未确定要住豪华型。'),
    ('withdrawal', '10870', (0, 1, 2), ((0, '500-1000元'), (0, '烤鱼豆腐')), (1,),
     '明确取消价格限制，仍要求有烤鱼豆腐。'),
    ('both_active', '10870', (0, 1, 2), ((0, '烤鱼豆腐'), (0, '芥末烤翅')), (0, 1),
     '最后一句仍明说两道菜，标注次序不表示替换。'),
    ('both_active', '8065', (12, 13, 14), ((12, '起泡酒'), (12, '牛里脊')), (0, 1),
     '找不到餐馆后仍询问哪家同时提供起泡酒和牛里脊。'),
    ('new_scope', '9041', (0, 1, 2), ((0, '京味爆肚'), (2, '馄饨')), (1,),
     '另找一家餐馆给母亲吃馄饨，前一项仍属于先前用餐任务。'),
    ('new_scope', '278', (0, 1, 2), ((0, '经济型'), (1, '舒适型')), (),
     '最后问的是景点，酒店的两个类型都不适用于当前任务。'),
    ('new_scope', '23', (0, 1, 2, 3, 4), ((0, '400-500元'), (4, '200元以上')), (1,),
     '预算分别属于酒店和后来推荐的景点，后者是当前查询。'),
    ('new_scope', '10712', (6, 7, 8, 9, 10), ((7, '房山区'), (10, '1000元以上')), (1,),
     '改为查餐馆的预算，原景点地址与当前餐馆筛选无关。'),
)
