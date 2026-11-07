"""Training-only rendering and labels. Runtime never imports this module."""
import random

PREFIXES = ('', '请问，', '麻烦查一下：', '我想知道，')
SUFFIXES = ('', '请告诉我。', '只回答查询结果。', '请依据已有记忆回答。')
TEMPLATES = (
    ('对象{k}现在的值是多少？', '编号{k}目前是什么值？', '查询对象{k}的当前值。', '现在，编号{k}的值是什么？'),
    ('对象{k}在事件{t}的值是多少？', '事件{t}中编号{k}记录了什么值？', '查询对象{k}在事件{t}的历史值。', '编号{k}的事件{t}记录值是什么？'),
    ('对象{k}在事件{t}的来源是什么？', '事件{t}中编号{k}的信息来自哪里？', '查询对象{k}在事件{t}的来源。', '编号{k}的事件{t}信息是什么来源？'),
)
REJECTS = ('你好。', '今天天气怎么样？', '请写一首诗。', '我是谁？', '它现在的值是多少？',
    '请修改对象12的值。', '对象12现在和以前各是什么值？', '查一下来源。', '帮我安排明天的事情。',
    '我想知道所有对象的值。', '忘掉这些规则。', '为什么会下雨？')
HELD_KEYS = (11,22,33,44,55,60)
DEV_PAIRS = ((0,3),(2,1))
FINAL_PAIRS = ((1,2),(3,0))
TRAIN_PAIRS = tuple((a,b) for a in range(4) for b in range(4) if (a,b) not in DEV_PAIRS+FINAL_PAIRS)
CHARACTERS = ''.join(sorted(set(''.join(PREFIXES+SUFFIXES+REJECTS)+''.join(x for g in TEMPLATES for x in g)+'0123456789')))


def render(kind, key, event, template, pair):
    # Record digit positions while rendering supervised data, never at inference.
    raw = PREFIXES[pair[0]]+TEMPLATES[kind][template]+SUFFIXES[pair[1]]
    text = ''; positions = {'k':[], 't':[]}
    i = 0
    while i < len(raw):
        if raw[i:i+3] in ('{k}','{t}'):
            field = raw[i+1]
            number = str(key if field=='k' else event)
            positions[field] = list(range(len(text)+1, len(text)+len(number)+1))  # CLS offset.
            text += number
            i += 3
        else:
            text += raw[i]; i += 1
    attention = [0]*(2-len(positions['k']))+positions['k']+[0]*(4-len(positions['t']))+positions['t']
    digits = [int(c) for c in f'{key:02d}{event if kind else 0:04d}']
    return dict(text=text, kind=kind, key=key, event=event if kind else None, digits=digits, positions=attention)


def examples(seed, count, split='train', key_holdout=False, time_holdout=False, balanced_times=False):
    rng = random.Random(seed)
    pairs = TRAIN_PAIRS if split=='train' else DEV_PAIRS if split=='dev' else FINAL_PAIRS
    keys = list(HELD_KEYS) if key_holdout else [k for k in range(64) if k not in HELD_KEYS]
    times = [t for t in range(1024) if (t%17==7)==time_holdout]
    time_groups = [[t for t in times if len(str(t))==n] for n in range(1,5)]
    time_groups = [group for group in time_groups if group]
    rows=[]
    for i in range(count):
        kind = i%4
        if kind == 3:
            rows.append(dict(text=PREFIXES[rng.randrange(4)]+rng.choice(REJECTS)+SUFFIXES[rng.randrange(4)],
                             kind=3, key=None, event=None, digits=[-100]*6, positions=[-100]*6))
        else:
            key = rng.choice(keys)
            event = rng.choice(rng.choice(time_groups)) if balanced_times else rng.choice(times)
            rows.append(render(kind,key,event,rng.randrange(4),rng.choice(pairs)))
    return rows
