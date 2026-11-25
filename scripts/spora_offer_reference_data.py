"""Paired reference and approval labels for two known dialogue candidates."""


TRAIN_NAMES = (
    (('北树会议安排', '北树'), ('烟波会议安排', '烟波')),
    (('晴山交通方案', '晴山'), ('蓝舟交通方案', '蓝舟')),
    (('青槐接待计划', '青槐'), ('檐灯接待计划', '檐灯')),
    (('溪口改版计划', '溪口'), ('草甸改版计划', '草甸')),
    (('小筑维修方案', '小筑'), ('春桥维修方案', '春桥')),
    (('菱角集会提议', '菱角'), ('芦苇集会提议', '芦苇')),
)
HELD_NAMES = (
    (('石阶放映计划', '石阶'), ('榆梢放映计划', '榆梢')),
    (('白船搬运方案', '白船'), ('乌鹊搬运方案', '乌鹊')),
    (('瓦松布置方案', '瓦松'), ('蒲叶布置方案', '蒲叶')),
)


# Target positions and intent are independent of the surface term used.
TRAIN_FORMS = (
    ('alias', '{a}就按这项办', '{a}的价格先说一下'),
    ('alias', '决定用{a}，其余条款照旧', '{a}是否提供额外服务'),
    ('pronoun', '那就按刚才讨论的这个来', '这项要准备多久'),
    ('pronoun', '这个我正式确认采用', '它具体有哪些注意事项'),
    ('other', '当前用另一项，刚才说的先算了', '刚才那项已经了解，我想问另一个'),
    ('other', '我改选另一个', '给我看剩下那个的细节'),
    ('ordinal', '采用{position}', '我想问{position}的细节'),
    ('ordinal', '{position}就定下了', '请再介绍一下{position}'),
    ('competing', '{b}只是了解，真正采用{a}', '提到{b}是举例，我问的是{a}的价格'),
    ('competing', '我不选{b}，已经决定{a}', '我先问{a}的细节，{b}暂时放在一边'),
    ('unbound', '', '先不用决定任何一个，再想想其他的办法'),
    ('unbound', '', '我们转个话题，我想问另一个完全不同的问题'),
)

HELD_FORMS = (
    ('alias', '拍板用{a}了，可以准备后面的工作', '请问{a}有什么附带条件'),
    ('pronoun', '刚谈过的这份我已经做了决定，就用它', '它这次费用会比预估高吗'),
    ('other', '眼下还是定下尚未细聊的那一项', '聊过的先放在这里，没聊的那项打算怎么做'),
    ('ordinal', '最终名单上只留{position}这套', '{position}的材料清单可以发过来吗'),
    ('competing', '我来问{b}只是为了比较，现在决定采用{a}', '{b}别填了，我只想听{a}的说明'),
    ('unbound', '', '这两个先都别动，我要重新想一想'),
    ('unbound', '', '换个任务，我想知道周一的安排'),
)


def scenes(forms, names, split):
    rows = []
    for family, (kind, agree, ask) in enumerate(forms):
        for index, pair in enumerate(names):
            for target in range(2) if kind != 'unbound' else (2,):
                focal = {'pronoun': target, 'other': 1 - target}.get(kind, (index + family + target) % 2)
                a, b = pair[target % 2][1], pair[1 - target % 2][1]
                history = (('usr', f'先解释一下{pair[focal][0]}。'),
                           ('sys', '刚才这项的细节我说明了，其他的方面也可以核对。'))
                for expected, template in ((True, agree), (False, ask)):
                    if template:
                        reply = template.format(a=a, b=b, position=('第一项', '第二项')[target % 2]) + '。'
                        rows.append(dict(key=f'{split}/{family}/{index}/{target}/{int(expected)}', family=kind,
                                         candidates=pair, history=history, reply=reply,
                                         target=target, approval=expected))
    return rows


def samples():
    return scenes(TRAIN_FORMS, TRAIN_NAMES, 'train'), scenes(HELD_FORMS, HELD_NAMES, 'held')


# Verified continuation, after the system offered two names. Index 2 marks a new topic.
REAL = (
    ('6319', 13, 0), ('10626', 13, 1), ('8595', 1, 1), ('8697', 1, 1),
    ('12027', 19, 1), ('7941', 9, 0), ('4304', 7, 1), ('8515', 13, 1), ('11382', 7, 1),
    ('8159', 15, 0), ('8159', 23, 0), ('7518', 1, 1), ('11659', 13, 0), ('2059', 1, 0),
    ('8979', 1, 2), ('7888', 1, 2), ('123', 1, 2),
)
