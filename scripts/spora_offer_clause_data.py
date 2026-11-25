"""Counterfactual offers with a reviewed decisive clause and target scope."""


SCENES = ('住处', '场地', '出行安排', '茶会布置', '印刷规格', '维修时段')
TRAIN_FORMS = (
    ('听起来行', '这回就采用这个备选', '我只想多了解一下，还没选定', '细节可以说一说吗'),
    ('好的，我听明白了', '按刚才提的备选走', '先把备选留在清单里，不要采用', '把后面的安排告诉我'),
    ('它的条件我看过了', '我同意采用它', '是否采用得等答复再定', '能把详情发给我吗'),
    ('这处和要求有些差别', '这点差别可以接受，现在就选它', '这个差别我接受不了，换别的吧', ''),
    ('我不需要额外服务', '刚说的备选我决定用了', '我仍想看另一项，先别选这个', ''),
    ('知道了', '请立即按它安排', '现在没到决定的时候', '还要核对其余要求'),
    ('', '确定要刚才推荐的那个', '只是问问那项的情况', '费用还能再核实一次吗'),
    ('这不是价格的问题', '我愿意把这份备选定下来', '我不打算改用这份备选', '原来的其余条件照旧'),
    ('你说的我都记下了', '把它作为最后的决定', '把它当作例子就好', ''),
    ('听着挺合适', '后面的事就以它为准', '等结果出来了我才决定要不要用它', ''),
    ('它看着不错', '不用再找了，我选它', '不用现在给我定下来', '再说一下注意事项'),
    ('', '我现在已经决定用你的建议', '如果合适我才会接受你的建议', '你有更细的信息吗'),
)

CALIBRATION_FORMS = (
    ('谢谢提醒', '目前就把这一项敲定下来', '目前还不确定是否采用这一项', '另外一件事先照旧'),
    ('你说得有道理', '我愿意选刚提出的方案', '等我比较之后再答复', '还有别的细节吗'),
    ('可以了解', '了解完了，就照它办', '只了解情况，这会儿先不要改变方案', ''),
    ('别加别的服务', '刚刚那项确实选中', '刚刚那项只考虑不确认', ''),
)

HELD_FORMS = (
    ('行，我听到了', '这就落定为刚才那套', '只登记成可选，今天不落定', '保留最初的附加事项'),
    ('好，先说联系方式', '联系方式可以慢慢找，这个我先确定采用', '我得先确认信息，仍不能说已经选了', ''),
    ('不需要取消原来的附加服务', '新方案本身就按你说的用', '新方案本身也还没同意', '其他维持原样'),
    ('看起来并不差', '我打算就这么处理', '我打算过两天再回复', '价格也请查一下'),
    ('', '把先前那项候补升级成本次的最终决定', '把先前那项候补留作比较就行', '其他条件仍一样'),
    ('你说的这个可以看看', '不用另选了，本次已经拍板', '仅供参考，别替我拍板', ''),
    ('听着能接受', '那本次明确选定它', '真要选还得看周末有没有空', '先记录下要核对的事情'),
    ('这一项听明白了', '可以正式纳入本次安排了', '请别写进本次已经确定的安排', ''),
    ('', '有点不方便，也就定这一个', '有点不方便，所以这次不确定它', '不相关的事情不用再讨论'),
    ('好，了解一下也行', '同时请把它设为最后采用的计划', '但这句话不表示我接受了它', ''),
)


def row(utterance, decisive, accepted, pair, group, target=None):
    if utterance.count(decisive) != 1:
        raise ValueError('Need a unique annotated decisive span')
    return {'dialogue': pair, 'index': int(accepted), 'group': group, 'reply': utterance, 'speaker': 'usr',
            'expected': accepted, 'target': target, 'anchor': decisive, 'pair': pair}


def polarity(forms, scenes, group):
    rows = []
    for index, (start, positive, negative, tail) in enumerate(forms):
        for scene in scenes if start else scenes[:1]:
            pair = f'{group}/{index}/{scene}'
            opening = f'关于{scene}，{start}' if start else ''
            for accepted, decisive in ((True, positive), (False, negative)):
                utterance = '。'.join(part for part in (opening, decisive, tail) if part) + '。'
                rows.append(row(utterance, decisive, accepted, pair, group))
    return rows


def scope(pairs, statements, group):
    rows = []
    for index, (first, second) in enumerate(pairs):
        for order, pattern in enumerate(statements):
            # Reverse the actual choice as well as the target across the set.
            accept, decline = (first, second) if order % 2 == 0 else (second, first)
            decisive = pattern.format(accept=accept, decline=decline)
            utterance = f'好，两项的内容都听到了。{decisive}。其他事情保持原样。'
            pair = f'{group}/{index}/{order}'
            rows.extend(row(utterance, decisive, approved, pair, group, target)
                        for target, approved in ((accept, True), (decline, False)))
    return rows


def samples():
    train = polarity(TRAIN_FORMS, SCENES, 'train_polarity')
    train += scope((('青坡案', '竹坛案'), ('远窗案', '红瓦案'), ('江桥案', '麦田案'), ('曲石案', '云岫案')),
                   ('我选{accept}，{decline}仅供比较', '不采用{decline}，这回确定用{accept}',
                    '{decline}是别人举的例子，我确定要{accept}'), 'train_scope')
    calibration = polarity(CALIBRATION_FORMS, ('借用物品',), 'cal_polarity')
    calibration += scope((('青篱案', '木门案'), ('桔灯案', '梨园案')),
                         ('最终定的是{accept}，先搁着{decline}',), 'cal_scope')
    held = polarity(HELD_FORMS, ('场次确认',), 'fresh_polarity')
    held += scope((('秋山案', '雪岭案'), ('栀木案', '黄藤案'), ('晚舟案', '藤桥案'), ('泊松案', '临溪案')),
                  ('{decline}值得问价，可是现在拍板的是{accept}',
                   '我问的是{decline}，同意采用的是{accept}'), 'fresh_scope')
    return train, calibration, held
