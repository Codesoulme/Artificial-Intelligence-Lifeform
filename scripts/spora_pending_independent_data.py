"""Held authorizations with fresh CrossWOZ episodes and contrastive replies."""


REAL_ACCEPT = (
    ('11281', 16), ('2487', 6), ('8811', 16), ('12333', 12), ('3580', 18), ('7812', 4),
    ('7523', 4), ('10246', 0), ('2470', 2), ('300', 8), ('3689', 0), ('12172', 10), ('6854', 6),
)
REAL_ASK = (
    ('11356', 0), ('4463', 10), ('2702', 4), ('12140', 8), ('10947', 18), ('8744', 12),
    ('9996', 6), ('2144', 0), ('5342', 20), ('3015', 22), ('9209', 0), ('11762', 4),
    ('9917', 12), ('3452', 0), ('11290', 0), ('10923', 12), ('12064', 0), ('10494', 10),
    ('6640', 4), ('7166', 4), ('9854', 6),
)
REAL_CONTINUE = (
    ('281', 2), ('11543', 8), ('9451', 12), ('12074', 2), ('8224', 2), ('11691', 8),
    ('1603', 14), ('8671', 2), ('8288', 10), ('8658', 2), ('229', 12), ('388', 10), ('11738', 8),
)


# Each pair receives exactly the same pending proposal, with opposite user decisions.
CONTRASTS = {
    'quotation': (
        ('原展位暂不可用，我提出改到东门。你现在要确认吗？',
         '小陈虽然没表态，我本人确定采用东门，请记下来。', '小陈刚说「确认采用东门」，我只是转述，还没决定。'),
        ('原路线施工，可以改走南桥。你愿意改吗？',
         '我同意改走南桥。请用这条覆盖旧路线。', '把「我同意改走南桥」这句话改得委婉一点，我没有授权。'),
        ('我们可改用一份更短的版本。要用它吗？',
         '我确认要短版，这是我现在的决定。', '我在练习说「我确认要短版」，当前仍不要改动。'),
        ('原主持人来不了，我提出由另一位接手。你确定采用吗？',
         '对，虽然上次的决定被人转述错了，这次我确实确认由你说的那位接手。',
         '会议纪要写了我会确认接手人。这句话只是在核对纪要，还没确认。'),
    ),
    'negation': (
        ('我提议将集合点改到东门。你同意吗？', '我同意东门，不同意继续用原地点。',
         '我不同意东门，同意继续用原地点。'),
        ('不能按旧时间进行，可以提前到八点。你要确认这个备选吗？',
         '不用保留旧时间，我确认采用八点。', '我确认的是可以查看备选，不是说我要改到八点。'),
        ('原材料不足，可以用替代材料。需要我把它设成最终方案吗？',
         '我不反对替代材料，就按你说的采用。', '我不反对继续问这种材料，先不要采用该方案。'),
        ('这个时段没空，我提议换到上午。现在确认吗？',
         '我确认换到上午，其余事项无需更改。', '我说的是无需更改。改到上午的提议先别确认。'),
    ),
    'condition': (
        ('我建议改用东门，但还要你确认门禁条件。现在确定吗？',
         '我已经核对门禁条件，东门确定采用。', '等我核对完门禁条件再决定，暂时不要采用东门。'),
        ('方案可以换成上午，需要你确定。', '上午有空这件事已经确认，改到上午就定了。',
         '假如上午有空才会改，现在我还不知道。'),
        ('车站是备选，原地点仍有效。要用车站吗？', '不用再等消息，我已决定用车站。',
         '先把车站放在列表，等消息出来我再决定。'),
        ('还有一个可以同时兼容的地方，你要确认更新吗？',
         '已核对，符合要求。我现在同意采用。', '如果符合要求我才同意，目前还没核对。'),
    ),
}


def scripted():
    for group, pairs in CONTRASTS.items():
        for index, (offer, positive, negative) in enumerate(pairs):
            for intent, reply in ((True, positive), (False, negative)):
                address = f'{group}/{index}/{int(intent)}'
                yield dict(key=address, family=group, passages=('我想看看有什么可改的安排。', offer, reply),
                           pending=True, spoken=intent, expected=intent, pair=f'{group}/{index}', binding='open')
    # The user actually accepts the older text. The supplied handle is invalid.
    yield dict(key='old/superseded', family='old', pending=True, spoken=True, expected=False, binding='superseded',
               passages=('上一次咱们提到改去东门。', '刚才的东门提议已经被替换，现在等你确认西门。',
                         '我翻到了旧消息，确认的就是之前那条东门。'))
    yield dict(key='old/expired', family='old', pending=False, spoken=True, expected=False, binding='expired',
               passages=('有人说过要换到东门。', '那条待确认已经过期，现在这项没有待决定的改动。',
                         '我这里还有此前的消息，我现在确认那份旧提议。'))
    yield dict(key='old/unbound', family='old', pending=True, spoken=True, expected=False, binding='unbound',
               passages=('先说一下有哪些备选。', '集合点和零食各有一个备选，分别都在等答复。',
                         '好的，确认。'))


# Selected before running the heads. These cases also pass through the memory backend.
TRANSACTION_KEYS = (
    'quotation/0/0', 'quotation/0/1', 'negation/0/0', 'negation/0/1', 'condition/0/0', 'condition/0/1',
    'old/superseded', 'old/expired', 'old/unbound',
)
