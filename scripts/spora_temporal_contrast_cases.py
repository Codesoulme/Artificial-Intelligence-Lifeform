"""Question families for separating version language from read intent."""

from spora.temporal_routing import CURRENT, LATER, PAST

ABSTAIN = 3

# Triples share the person and field. The selected version is the variable.
VERSION_TRIPLES = (
    ('眼下登记给{name}的{field}是哪项？', '这次修订前给{name}登记的{field}是哪项？',
     '这次修订后给{name}登记的{field}是哪项？'),
    ('一律以最新版为准，{name}的{field}填的是什么？', '一律以旧版为准，{name}的{field}填的是什么？',
     '一律以修订刚完成的版本为准，{name}的{field}填的是什么？'),
    ('更正过了，麻烦查{name}现在的{field}。', '更正过了，麻烦查{name}原本的{field}。',
     '更正过了，麻烦查{name}改完那一刻的{field}。'),
    ('{name}用的是这版{field}的哪项？', '{name}用的是上版{field}的哪项？',
     '{name}在改版之后采用的{field}是哪项？'),
    ('别看早先那页，{name}现用的{field}是什么？', '别看眼下那页，{name}还没改时的{field}是什么？',
     '别看更正前那页，更正一结束{name}的{field}改成了什么？'),
    ('{name}最近留的{field}是哪一个？', '{name}最早留的{field}是哪一个？',
     '{name}换过之后留的{field}是哪一个？'),
)
NON_READ = (
    '我知道{name}改了{field}，现在先别读记录。',
    '把{name}原来的{field}覆盖掉，先不用告诉我原值。',
    '把「{name}从前选择的{field}是什么」改成问候语。',
    '什么时候修改了{name}的{field}？我只问日期。',
    '帮我猜{name}下次会怎么选{field}。',
    '上一次{field}让我想起{name}，请帮我写个段子。',
)
CAL_TRIPLES = (
    ('以手头这一份为准，{name}的{field}是哪样？', '翻未经更改的原件，{name}的{field}是哪样？',
     '翻更改刚生效的那一份，{name}的{field}是哪样？'),
    ('只看仍然有效的，{name}登记的{field}是什么？', '只看已经作废的，{name}那时登记的{field}是什么？',
     '只看刚被替换上来的，{name}那时登记的{field}是什么？'),
)
CAL_NON_READ = (
    '把{name}的{field}如何变更写成一个虚构例子。',
    '别碰{name}现用的{field}，我只是想算一笔账。',
    '把{name}当前的{field}换成新值，原值无需显示。',
)
HELD_TRIPLES = (
    ('{name}那张还在用的卡上，哪样写在{field}一栏？',
     '改动还没做的时候，{name}的{field}栏填的哪样？',
     '改动一落地，{name}的{field}栏就换成哪样了？'),
    ('别给废掉的，照有效条目说{name}选了哪种{field}。',
     '就说换掉的那项，{name}原来选的是哪种{field}？',
     '别给更早的，{name}的{field}第一次修订生效时是什么？'),
    ('按今天能执行的版本，{name}要的是哪款{field}？',
     '按当时尚未重写的版本，{name}要的是哪款{field}？',
     '按重写一完成时的版本，{name}要的是哪款{field}？'),
)
HELD_NON_READ = (
    '不要打开{name}的资料。顺便解释一下{field}这个词。',
    '请问{name}为什么更换{field}，别复述更换前后两项。',
    '请把{name}上次怎么选{field}这一句翻成英文。',
    '给{name}的{field}提几条从没用过的新点子。',
    '先把{name}的{field}改一下，回头再读。',
)


def temporal_rows(names, fields, triples, non_read):
    rows = []
    for index, name in enumerate(names):
        field = fields[index % len(fields)]
        for triple in triples:
            rows.extend((template.format(name=name, field=field), target)
                        for template, target in zip(triple, (CURRENT, PAST, LATER), strict=True))
        rows.extend((template.format(name=name, field=field), ABSTAIN) for template in non_read)
    return rows
