"""Status supervision for values that occur in earlier dialogue turns."""

from enum import IntEnum

from scripts.audit_spora_revision_choices import PAIRS
from scripts.spora_revision_rank_data import FRESH, FRESH_TERMS, SCENES, training_pairs


class Status(IntEnum):
    ACTIVE = 0
    SUPERSEDED = 1
    MENTION = 2


def contrast_rows(pairs, values, inactive_statuses):
    rows = []
    for (name, history, endings, _), terms, inactive in zip(pairs, values, inactive_statuses, strict=True):
        rows.append((name, (*history, endings[0]), terms, (Status.ACTIVE, inactive)))
        rows.append((name, (*history, endings[1]), terms, (Status.SUPERSEDED, Status.ACTIVE)))
    return tuple(rows)


def training_rows():
    primary = training_pairs()
    roles = (Status.MENTION, Status.SUPERSEDED, Status.MENTION,
             Status.MENTION, Status.SUPERSEDED, Status.MENTION)
    rows = list(contrast_rows(primary, [(old, other) for _, _, old, other, _ in SCENES for _ in roles],
                              roles * len(SCENES)))
    for name, topic, old, other, unchanged in SCENES:
        rows.append((name + '取消', (f'{name}的{topic}用{old}，记得{unchanged}。', f'曾想换成{other}。',
                                     f'撤销这次对{topic}的要求，两项都不用。'),
                     (old, other), (Status.SUPERSEDED, Status.MENTION)))
        rows.append((name + '引用', (f'讨论案例里的{topic}是{other}。', f'{old}只是另外一例。',
                                     f'这两种说法都只是举例，先别确定{topic}。'),
                     (old, other), (Status.MENTION, Status.MENTION)))
        rows.append((name + '逆序', (f'先听到备选{other}，还没有决定。', f'{name}的{topic}原定{old}，记得{unchanged}。',
                                     '恢复原定那项，先不用备选。'),
                     (old, other), (Status.ACTIVE, Status.MENTION)))
        rows.append((name + '逆序', (f'先听到备选{other}，还没有决定。', f'{name}的{topic}原定{old}，记得{unchanged}。',
                                     '采用之前听到的备选，取消原定那项。'),
                     (old, other), (Status.SUPERSEDED, Status.ACTIVE)))
    return tuple(rows)


ARCHIVE = contrast_rows(PAIRS, (('胶带', '纱布'), ('夜航图', '潮汐'), ('周三', '周四'), ('南厅', '北厅'),
                                ('灯亮', '灯灭'), ('若松', '安桥'), ('春潮', '青苔'), ('竹台', '涧桥')),
                       (Status.SUPERSEDED, Status.MENTION, Status.SUPERSEDED, Status.MENTION,
                        Status.MENTION, Status.SUPERSEDED, Status.SUPERSEDED, Status.MENTION))
HELD = contrast_rows(FRESH, [terms[:2] for terms in FRESH_TERMS],
                     (Status.MENTION, Status.MENTION, Status.MENTION, Status.SUPERSEDED,
                      Status.SUPERSEDED, Status.MENTION, Status.MENTION, Status.MENTION))
NULL = (
    ('挂画全部撤回', ('前厅先挂《花房》，其他画放库里。', '后来在考虑将《码头》挂到前厅。', '今天前厅不挂画，两张都收起。'),
     ('花房', '码头'), (Status.SUPERSEDED, Status.MENTION)),
    ('票样只是转述', ('展览的往期票样上写着秋禾节。', '征集时有人用秋石节举例。', '这些是例文，展览名称还没定。'),
     ('秋禾节', '秋石节'), (Status.MENTION, Status.MENTION)),
    ('临时教室取消', ('小组上课最初去环形厅。', '中午改到蓝帘室，时间未变。', '房间还需确认，这两个安排都作废。'),
     ('环形厅', '蓝帘室'), (Status.SUPERSEDED, Status.SUPERSEDED)),
    ('食物仅是样例', ('我拿藕片举个样例，当时并未订菜。', '另有一个样例用海带。', '先保存范文，今天的菜稍后再说。'),
     ('藕片', '海带'), (Status.MENTION, Status.MENTION)),
    ('撤去口号', ('屏幕原先加“看这里”。', '“向前看”是讨论过的替代说法。', '屏幕去掉口号，替代说法也不启用。'),
     ('看这里', '向前看'), (Status.SUPERSEDED, Status.MENTION)),
    ('色卡未审批', ('蓝灰是供大家讨论的色卡示例。', '另一张示例选了土黄。', '两张都没下单，选色要等审批。'),
     ('蓝灰', '土黄'), (Status.MENTION, Status.MENTION)),
)
