"""Chinese time candidates in confirmed, superseded and unresolved requests."""

from dataclasses import dataclass
import random
import re

from scripts.spora_memory_composition_cases import HELD


@dataclass(frozen=True)
class Example:
    family: str
    text: str
    current: str | None


TRAIN_TIMES = (
    '周四早上', '周一中午', '今晚六点', '明早七点', '后天上午', '今日傍晚',
    '星期六下午', '下周二晚上', '明天下午两点', '周末晚上', '早上八点半', '上午十点',
    '周六九点', '周四晚八点', '周二清晨', '次日上午',
)
TRAIN_FORMS = (
    ('confirmed', '请给程怀写一条会面消息，时间定在{a}，按登记地点碰头。', 'a'),
    ('confirmed', '{a}可以见，按这条替我写通知。', 'a'),
    ('revision', '原本约在{b}，现在换到{a}，请写给对方。', 'a'),
    ('revision', '以{a}为准，早先说的{b}作废，写个简短通知。', 'a'),
    ('revision', '{b}和{a}原来都提到，最后定了{a}，地点别动。', 'a'),
    ('revision', '记下{b}曾经考虑过，正式约的是{a}，把最新安排写出来。', 'a'),
    ('quote', '别把旧通知里的“{b}”抄进来，今天敲定的是{a}。', 'a'),
    ('quote', '按{a}写，文中的“{b}”只是示例。', 'a'),
    ('revision', '我确定{a}来取书；{b}只留作备选。', 'a'),
    ('none', '还没定。有人提议{a}或{b}，先不发通知。', None),
    ('none', '示例句子是“{a}碰头”，这次还没有定时间。', None),
    ('none', '{a}取消了，暂时没有新方案。', None),
    ('none', '如果{a}能安排我再确认，别当成已经约好。', None),
    ('none', '帮我解释“{a}”这几个字的用法。', None),
    ('none', '我没看日程，现在只要告诉我见面地点是否已经登记。', None),
)


def training():
    rng = random.Random(71502)
    for family, template, selected in TRAIN_FORMS:
        for a, b in zip(TRAIN_TIMES, rng.sample(TRAIN_TIMES, len(TRAIN_TIMES)), strict=True):
            if a != b:
                yield Example(family, template.format(a=a, b=b), a if selected else None)


HELD_TEXTS = (
    Example('revision', '刚和阿榆说定星期三傍晚。此前群里讲的周日中午不算数，照最新场次写。', '星期三傍晚'),
    Example('revision', '别再照着今晚九点了，改成明早十点，再把消息发给对方。', '明早十点'),
    Example('revision', '场地不用改。周五傍晚那个时候来不了，见面的点儿已经换成下周三上午。', '下周三上午'),
    Example('revision', '周日清早才是这次的碰头时间；聊天里留着的周五下午是前一版。', '周日清早'),
    Example('three', '周五晚上提过，星期日中午也讨论过，最后拍板的是星期三中午，请替我转告。', '星期三中午'),
    Example('three', '新通知写后天晚上。明天上午和周五中午都只是讨论记录。', '后天晚上'),
    Example('quote', '同事发来的参考开头是“今晚七点你方便吗”，只参考语气，本次定在明天下午三点。', '明天下午三点'),
    Example('quote', '周五傍晚已经定下，后面引的那句“明早十点取东西”是另一件事。', '周五傍晚'),
    Example('backup', '请按今晚九点写聚会地点，明天早上只作无法赴约时的替代。', '今晚九点'),
    Example('backup', '还没有决定用哪个时间，星期三中午、周日清早都可能。先不要发。', None),
    Example('quote_only', '试着把“明早十点去取”译成英文，只给译文。', None),
    Example('quote_only', '他原话是“周五傍晚再说”，我想知道这种说法礼不礼貌。', None),
    Example('canceled', '之前的今晚九点已经取消。什么时候重新约，我还不知道。', None),
    Example('conditional', '要是明早十点有人开门，我们可能那时候见，现在先别确定。', None),
    Example('conditional', '如果场地通过审核，再考虑星期三中午，目前不做约定。', None),
    Example('unrelated', '周五傍晚之前天气都不错，我想问登记的饮品还剩多少。', None),
    Example('no_time', '见面日期暂缺，别编一个出来，告诉我还需要哪条信息。', None),
    Example('no_time', '帮我给朋友解释为什么纸巾能吸水，用一句话。', None),
)


def held_texts():
    for spec in HELD:
        options = [('single', spec.scenario.question, spec.time_pattern)]
        if spec.changed_question:
            options.append(('single_changed', spec.changed_question, spec.changed_pattern))
        for family, text, pattern in options:
            yield Example(family, text, re.search(pattern, text).group())
    yield from HELD_TEXTS
