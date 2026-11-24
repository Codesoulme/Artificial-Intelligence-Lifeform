"""Produce training and calibration turns with misleading old or unchanged values."""

from random import Random

from scripts.train_spora_dialogue_working_state import annotated
from spora.research_dialogue_working_state import FIELDS, TURN_LIMIT, characters


ALIASES = {
    'time': ('约好的钟点', '哪天几点', '到场的时段'),
    'place': ('碰面位置', '那边的地址', '过去的地方'),
    'person': ('负责的人', '经手的姓名', '联络的那位'),
    'quantity': ('实际份量', '货品件数', '请留的数目'),
    'gear': ('要带的物品', '那一项内容', '原先选的物件'),
}

INITIAL = (
    '{topic}这件事，{a}先填{v0}，{b}那项由{v1}对应，另记{c}为{v2}。',
    '有条{topic}记录：{b}定的是{v1}。{a}用{v0}，{c}写{v2}。',
    '我把{v2}记在{c}一栏，{a}是{v0}。至于{b}，填{v1}。',
)
UPDATE = (
    '至于{a}，上次说{old}，后来改成{value}。{b}还是{unchanged}。',
    '这次{topic}别沿用{old}。实际的{a}写{value}，原定{b}继续是{unchanged}。',
    '把{value}替换进去，填的是{a}。{unchanged}对应的{b}保持原先那样。',
    '{b}不调整，仍用{unchanged}。需要变动的{a}已经确定为{value}。',
    '{old}作废。{topic}的{a}接下来按{value}算；{b}不变。',
    '重新确认了一遍，{topic}这次是{value}。这是新{a}，之前的{old}不用。'
)
CAL_INITIAL = '我核对了{topic}的三栏：{c}录入{v2}；{a}录入{v0}，剩下{b}是{v1}。'
CAL_UPDATE = '更新通知到了，{b}延用{unchanged}，而{a}以{value}为准。{old}是上次记录的。'
NO_WRITE = (
    '{topic}的{a}我可能要改，但现在没定。先沿用{old}可以吗？',
    '先别提交，{value}只是我想到的一个备选{a}，原来的{old}还有效。',
    '能把{topic}的{a}和{b}再发给我？我只看一下。',
)


def label_text(template, phrases, spans):
    """Insert field-tagged placeholders and extract character supervision."""
    text, locations = annotated(template, phrases)
    for field, marker in spans.items():
        lo = text.index(marker)
        if text.count(marker) != 1:
            raise ValueError('A value marker must appear exactly once')
        locations[FIELDS.index(field)] = (lo + 1, lo + len(marker))
    if not 0 < len(text) <= TURN_LIMIT:
        raise ValueError('Revision example exceeds supported turn length')
    return text, locations


def dialogue_samples(specs, fixture, names, vocabulary, *, calibration=False):
    rng = Random(26170 + int(calibration))
    rows, negative = [], []
    templates = (CAL_UPDATE,) if calibration else UPDATE
    eligible_init = (CAL_INITIAL,) if calibration else INITIAL

    def draw(field):
        value = rng.choice(vocabulary[field])
        if field == 'time':
            value = rng.choice(vocabulary['day']) + value
        return value

    for spec in specs:
        domain = spec['case']
        fields = spec['fields']
        for offset in range(6 if calibration else 24):
            cycle = offset % len(fields)
            a, b, c = fields[cycle:] + fields[:cycle]
            order = (a, b, c)
            full_names = dict(zip(fields, names[domain], strict=True))
            aliases = {field: (full_names[field], *ALIASES[field])[(offset // 3) % 4] for field in order}
            initial = {field: draw(field) for field in order}
            # Require distinct strings so corrections and retained values can be scored exactly.
            attempts = 0
            while len(set(initial.values())) != len(order):
                initial = {field: draw(field) for field in order}
                attempts += 1
                if attempts >= 16:
                    raise ValueError('Could not form distinct observed values')
            value = draw(a)
            while value in initial.values():
                value = draw(a)
            topic = fixture[domain]['label']
            header = {'topic': topic, **dict(zip('abc', (aliases[key] for key in order), strict=True))}
            first = label_text(eligible_init[(offset // 3) % len(eligible_init)],
                               {**header, **{f'v{i}': initial[field] for i, field in enumerate(order)}}, initial)
            words = {**header, 'old': initial[a], 'value': value, 'unchanged': initial[b]}
            update = label_text(templates[(offset // 3) % len(templates)], words, {a: value})
            question = f'{topic}的{aliases[a]}、{aliases[b]}和{aliases[c]}分别登记了什么？'
            rows.append({'first': first, 'update': update, 'question': question,
                         'initial': initial, 'change': {a: value}})
            if calibration:
                words = dict(words)
                text = '关于{topic}，我只是在考虑换{a}，{value}暂作备用，请继续保持原安排。'.format_map(words)
                negative.append(text)
            elif offset % 3 == 0:
                negative.append(NO_WRITE[(offset // 3) % len(NO_WRITE)].format_map(words))
    return rows, negative
