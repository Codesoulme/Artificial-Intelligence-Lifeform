"""Independent dialogue turns for the isolated proposal transaction replay."""


# First user turn, assistant turn, current user turn and whether this reply approves
# the particular proposal being discussed. Handle validity is checked separately.
REPLIES = {
    'info': ('把集合点的选项先说给我听。', '我提议改成银杏亭，你是否决定采用？',
             '好的，先告诉我这里的具体位置，我还没有决定。', False),
    'accept_snack': ('有没有另一种零食可选？', '现在可以选薄荷糖，我给你留了备选。你要确认吗？',
                     '可以，就用薄荷糖好了，请保存。', True),
    'accept_place': ('原约定地点需要调整，有什么建议？', '我提出把集合点改成银杏亭。现在决定用它吗？',
                     '行啊，就改去银杏亭。请按这个记下来。', True),
    'accept_changed': ('计划有变化，要另找个地点。', '建议用车站代替原地点，你现在确认吗？',
                       '会更方便，就用车站吧，我已经确定了。', True),
    'ambiguous': ('先把两个事项的备选都留下。', '一个关于集合点，另一个关于零食，两个都等你答复。',
                  '好的，那就照你说的办。', False),
    'other_topic': ('有个备选地点我还在考虑。', '我已把备选放在待确认里，你同意采用吗？',
                    '好的，换个事。你知道下周还有什么活动吗？', False),
    'again': ('刚才已讨论集合地点。', '我已经说明了当前集合点。这里没有新的更正提案。',
              '收到，请再给我说一下地址。', False),
}
