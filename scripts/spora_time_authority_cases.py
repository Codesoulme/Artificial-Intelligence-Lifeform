"""Paired Chinese time mentions with different levels of confirmation."""

from dataclasses import dataclass
import random

from scripts.spora_active_time_cases import Example


CONFIRMED, UNSET, UNRELATED = range(3)
ACTIVE, RETIRED, TENTATIVE, QUOTED = range(4)


@dataclass(frozen=True)
class Labeled:
    example: Example
    intent: int
    roles: tuple[tuple[str, int], ...]


TRAIN_TIMES = (
    '周一上午', '周四傍晚', '今晚八点', '明早六点', '后天中午', '今日下午',
    '星期六早上', '下周一晚上', '明天下午两点', '周末早上', '清晨六点', '上午十一点',
    '周六十点', '周四晚九点', '周二中午', '次日下午',
)
# The same values appear in both active and inactive roles, independently of their order.
FORMS = (
    ('confirmation', '跟唐桐说一声，我们就定{a}见面，写一句他能收到的消息。', CONFIRMED, ACTIVE, None),
    ('confirmation', '请直接发个成品。地点按登记，关于时间，我确定{a}。', CONFIRMED, ACTIVE, None),
    ('correction', '我把先前的{b}撤掉了，确认{a}会合。请给我改好的短讯。', CONFIRMED, ACTIVE, RETIRED),
    ('correction', '{a}才是我们敲定的档期。公告上还留着{b}，那版早作废了。', CONFIRMED, ACTIVE, RETIRED),
    ('correction', '新安排是{a}，不是原通知写的{b}。给大家写最新方案。', CONFIRMED, ACTIVE, RETIRED),
    ('correction', '原话中有{b}，那个时间不生效；请采用双方已经同意的{a}。', CONFIRMED, ACTIVE, RETIRED),
    ('quote', '安排确认在{a}。草稿“{b}附近见”供你参考句式，时间别照抄。', CONFIRMED, ACTIVE, QUOTED),
    ('quote', '对方拿“{b}取物”举例。我已经确认{a}，请写可以直接发的正文。', CONFIRMED, ACTIVE, QUOTED),
    ('backup', '我们取{a}这一档，{b}留给改约时再考虑。现在只写确定的时间。', CONFIRMED, ACTIVE, TENTATIVE),
    ('backup', '{b}是备用档，目前实际要用{a}，帮我发个集合提醒。', CONFIRMED, ACTIVE, TENTATIVE),
    ('conditional', '若场地同意才考虑{a}，目前还在等消息，暂时没有定下。', UNSET, TENTATIVE, None),
    ('conditional', '预订条件还没满足。{a}只是可能用到，别写成确定。', UNSET, TENTATIVE, None),
    ('ambiguous', '{a}和{b}都有人提议，双方还没做决定。', UNSET, TENTATIVE, TENTATIVE),
    ('canceled', '我把跟朋友约的{a}取消了。另约什么时候还没谈好。', UNSET, RETIRED, None),
    ('canceled', '原先{a}碰头已经作废，别再按这条安排。新时间空着。', UNSET, RETIRED, None),
    ('canceled', '从安排表上删去{a}，这是唯一需要改的地方。等我再说。', UNSET, RETIRED, None),
    ('quote_only', '把“{a}见面”翻成英语，直接给译文。', UNRELATED, QUOTED, None),
    ('quote_only', '“{a}取东西”听起来客气吗？这是书上的句子。', UNRELATED, QUOTED, None),
    ('quote_only', '请把对方说的“{a}会合”读给我听，我没有要约人。', UNRELATED, QUOTED, None),
    ('unrelated', '{a}以前天气不错，我想问的是桌上还有几支笔。', UNRELATED, QUOTED, None),
    ('unrelated', '顺便提到{a}，我那时有课。现在帮我解释为什么树影会移动。', UNRELATED, QUOTED, None),
    ('unresolved', '新安排还没给出时间，先别确定会面。', UNSET, None, None),
    ('unrelated', '帮我把木楼梯响的几个可能原因说清楚。', UNRELATED, None, None),
)


def training_pairs():
    rng = random.Random(82562)
    used = set()
    for family, template, intent, role_a, role_b in FORMS:
        for a, b in zip(TRAIN_TIMES, rng.sample(TRAIN_TIMES, len(TRAIN_TIMES)), strict=True):
            text = template.format(a=a, b=b)
            if a == b or text in used:
                continue
            used.add(text)
            roles = tuple((time, role) for time, role in ((a, role_a), (b, role_b)) if role is not None)
            yield Labeled(Example(family, text, a if intent == CONFIRMED else None), intent, roles)


FRESH = (
    Example('confirmed', '钟意那边答应了，明晚九点见。替我拟条简讯告诉其他人。', '明晚九点'),
    Example('quote_only', '校对这句话的标点：“明晚九点见”。内容原样保留。', None),
    Example('conditional', '明晚九点只是有空才可能约，我的行程还没出来。', None),
    Example('canceled', '把明晚九点从那条邀请上划掉，改约时间我还在等。', None),
    Example('confirmed', '把这场约在下周五下午三点，我和对方已经都同意。', '下周五下午三点'),
    Example('quote_only', '给这句配拼音：下周五下午三点能来吗？', None),
    Example('conditional', '下周五下午三点要等借到钥匙才能用，目前借不到。', None),
    Example('canceled', '下周五下午三点那档排不上了，会面改天，回头再敲时间。', None),
    Example('confirmed', '可以通知大家了，星期三早晨开始搬书。', '星期三早晨'),
    Example('quote_only', '检查“星期三早晨开始搬书”的英文有无语病。', None),
    Example('conditional', '星期三早晨先作为备选，我必须等值班表公布后才知道。', None),
    Example('canceled', '星期三早晨停办，不转到任何新的场次。', None),
    Example('correction', '两个人都接受周日九点。还写在表上的周二清早已经失效，麻烦改通知。', '周日九点'),
    Example('correction', '周日九点那个会面已移走，现在请用周二清早安排。', '周二清早'),
    Example('backup', '周日九点、周二清早二选一，他们还没有回复。', None),
    Example('unrelated', '周日九点和周二清早通常哪时候更冷？我只想知道天气。', None),
    Example('correction', '下午四点是讨论过的旧方案。最终敲定的是后天清晨。', '后天清晨'),
    Example('correction', '后天清晨那版一早就取消了，确定采用下午四点。', '下午四点'),
    Example('backup', '后天清晨是首选，下午四点是没约上才用的备用档，双方已确认首选。', '后天清晨'),
    Example('quote_only', '对比“后天清晨”和“下午四点”这两个短语的构成。', None),
    Example('cancel_only', '不用约了，等我处理完手边的事。', None),
    Example('unrelated', '告诉我蒸馒头为什么会膨胀，想听个简单解释。', None),
)
