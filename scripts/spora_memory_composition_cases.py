"""Training invitations and held-out ordinary replies with internal values."""

from dataclasses import dataclass

from scripts.audit_spora_memory_in_ordinary_replies import CASES, Case


TRAIN_NAMES = ('纪岚清', '陶书铭', '常言溪', '林槿安', '叶逸秋', '江闻雪')
TIMES = ('今天下午', '今晚七点', '周日早上', '明早八点', '周二中午', '周五傍晚')
FIELDS = {
    'places': ('水杉厅', '杏花巷', '南榆阁', '松涛廊', '杏林馆', '东石台'),
    'items': ('黑雨衣', '金属夹', '白手套', '绿水壶', '纸风筝', '银钥匙'),
    'books': ('潮声旧梦', '霜花集', '夜雨初晴', '梅影书简', '北雁归来', '渡河故事'),
    'drinks': ('姜茶', '酸梅汁', '冰牛乳', '桂花露', '糯米浆', '橙皮茶'),
}
# Each request keeps the values out of the visible prompt. The reply formats are
# authored independently of held wording and expose both memory slot orders.
FORMS = (
    (('places', 'items'),
     '帮我给{name}发条短信，约{time}在登记的地点碰面，告诉对方把登记的用品带来。',
     ('{time}在', 0, '碰面，麻烦带上', 1, '。')),
    (('items', 'places'),
     '替我写给{name}的会面消息。时间是{time}，要把记录中的用品带来，地点也从记录查。',
     ('请带上', 0, '，{time}我们在', 1, '见面。')),
    (('books', 'drinks'),
     '写条消息给{name}，说{time}她挑的书已经备好，我会带上她平时选的饮料。',
     ('{time}你要的《', 0, '》已经备好，我会带上', 1, '。')),
    (('drinks', 'books'),
     '请对{name}说，{time}可以取已登记的那本书，并告诉她我会给她带登记的饮品。一句话。',
     ('{time}可以来取《', 1, '》，我会给你带', 0, '。')),
    (('places', 'books'),
     '给{name}写一句能发的通知，请他把已经登记的书带到记下的碰头地点，{time}见。',
     ('{time}我们在', 0, '见，请带上《', 1, '》。')),
    (('books', 'places'),
     '用一句话约{name}{time}见，地点按已登记约定，提醒对方带已登记书目中的书。',
     ('{time}请带上《', 0, '》到', 1, '，我们在那里见面。')),
)


@dataclass(frozen=True)
class Held:
    scenario: Case
    time_pattern: str
    action_groups: tuple[str, ...]
    changed_question: str = ''
    changed_pattern: str = ''


HELD = (
    Held(CASES[0], '周五傍晚|星期五傍晚', ('见|碰面', '带'),
         CASES[0].question.replace('周五傍晚', '星期三中午'), '周三中午|星期三中午'),
    Held(CASES[1], '明[天日]?[早晨]八点|明天[早上]+8点|明早8点|明天.{0,3}8:00', ('见|碰面', '带'),
         CASES[1].question.replace('明早八点', '明早十点'), '明[天日]?[早晨]十点|明早10点|明天.{0,3}10:00'),
    Held(CASES[2], '下午', ('书.{0,12}(?:备好|准备好)|备好|准备好', '带')),
    Held(CASES[3], '今晚七点|今天[晚夜].{0,3}七点|今晚7点|今晚.{0,3}19:00', ('见|碰面', '带'),
         CASES[3].question.replace('今晚七点', '今晚九点'), '今晚九点|今晚9点|今天[晚夜].{0,3}九点|今晚.{0,3}21:00'),
    Held(
        Case('岑婉听', ('places', 'items'), (('新柳园', '杏雨台'), ('细绳圈', '皮笔筒')),
             '我想约岑婉听周日早上碰头。地点和该带的东西按登记过的安排，帮我写好一句邀约。'),
        '周日早上|星期日早上|星期天早上|周日早晨', ('见|碰头|碰面', '带'),
    ),
    Held(
        Case('黎书然', ('drinks', 'books'), (('莓果茶', '核桃乳'), ('月下小站', '芦岸晨歌')),
             '帮我给黎书然回一句，她挑的书给她留着，周二中午可以取。我还想捎上她常点的饮料，两样都请写名字。'),
        '周二中午|星期二中午', ('取|拿|领|留', '带|捎'),
    ),
)
